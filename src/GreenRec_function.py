import faiss
from pympler import asizeof
import numpy as np
import cupy as cp
from scipy.optimize import minimize

class GreenRecRecomender():
    class userMeta():
        class KalmanFilter():
            def __init__(self, num_dim):
                self.K = num_dim
                self.preferV = cp.zeros((self.K,), dtype=cp.float32)
                self.F = cp.eye(self.K, dtype=cp.float32)
                self.H = cp.eye(self.K, dtype=cp.float32)
                self.P = cp.eye(self.K, dtype=cp.float32)
                self.Q = cp.eye(self.K, dtype=cp.float32) * 0.01
                self.R = cp.eye(self.K, dtype=cp.float32) * 0.1
                self.num_update = 0
            
            def predict(self):
                self.preferV = cp.dot(self.F, self.preferV)
                self.P = cp.dot(cp.dot(self.F, self.P), self.F.T) + self.Q
            
            def update(self, measurement):
                measurement = cp.asarray(measurement, dtype=cp.float32)  # GPU로 옮기기
                y = measurement - cp.dot(self.H, self.preferV)  # 측정 잔차
                S = cp.dot(self.H, cp.dot(self.P, self.H.T)) + self.R  # 잔차 공분산
                K = cp.dot(cp.dot(self.P, self.H.T), cp.linalg.inv(S))  # 칼만 이득

                self.preferV = self.preferV + cp.dot(K, y)
                I = cp.eye(self.K, dtype=cp.float32)
                self.P = cp.dot((I - cp.dot(K, self.H)), self.P)

            def get_preference(self, vecTS):
                vecTS = cp.asarray(vecTS, dtype=cp.float32)  # GPU로 옮기기
                T = vecTS.shape[0]

                if self.num_update:
                    self.predict()
                    self.update(vecTS[-1])
                else:
                    for t in range(T):
                        measurement = vecTS[t]  # 현재 시점의 임베딩 벡터
                        self.predict()
                        self.update(measurement)
                
                self.num_update += 1
                return cp.asnumpy(self.preferV) # CPU로 복사하여 반환
            
        def __init__(self, id):
            self.id = id
            self.score = None
            self.relative_score = None

            self.kf = self.KalmanFilter(64)


    def __init__(self, dim, param):
        self.input_dim = dim
        self.param = param
        self.index = 0
        self.index2id = {}

        self.userDB = {}

        self.CHR_target = 0
        self.HR_target = 0
        self.theta_base = 0

        self.nowRR = 0
        self.nowCH = 0
        self.nowSessionRR = 0

        hnsw_vec_db = faiss.IndexHNSWFlat(dim, param, faiss.METRIC_INNER_PRODUCT)
        hnsw_vec_db.hnsw.efConstruction = 32
        hnsw_vec_db.hnsw.efSearch = 16
        self.vecDB = faiss.IndexIDMap2(hnsw_vec_db)


    def predictPreference(self, uid, vecTS):
        preferV = self.userDB[uid].kf.get_preference(vecTS) #preferV 업데이트

        return preferV

    
    def cacheTargetChange(self, new_CHR, new_HR, nowRR, baseTheta):
        self.nowCH = 0
        self.nowRR = 0
        self.nowSessionRR = nowRR
        self.theta_base = baseTheta
        
        if(new_CHR is None):
            return self.CHR_target, self.HR_target
        else:
            self.CHR_target = new_CHR
            self.HR_target = new_HR
            return self.CHR_target, self.HR_target

    #cache user key
    def metaRegister(self, uid):
        if(uid in self.userDB.keys()):
            return
        else:
            self.userDB[uid] = self.userMeta(uid)


    #preference vector based aware approx. matching
    ###############################################################
    def cacheLookup_preferApprox(self, uid, log, prefer_emb):
        #1. CHR 범위를 구함
        #2. 현재 CHR이 범위 안에 있는지 검사 (범위 밖이라면 임계값 제어)
        #3. 캐시 검색 후 hit/miss

        def getCHRTolerance(time, init_range=0.5, epsilon=0.001, alpha=0.1):
            sigmoid_tolerance = epsilon + (init_range - epsilon) / (1 + np.exp(alpha * (time - self.nowSessionRR / 2)))
            log_tolerance = epsilon + (init_range - epsilon) * (np.log(self.nowSessionRR + 1) - np.log(time + 1)) / np.log(self.nowSessionRR + 1)
            k = -np.log(epsilon / init_range) / self.nowSessionRR
            exp_tolerance = init_range * np.exp(-k * time)

            return log_tolerance
        
        #get CHR_tolerance range
        CHR_tolerance = getCHRTolerance(self.nowRR)
        print(f"CHR range({CHR_tolerance}): {max(0, self.CHR_target - CHR_tolerance)} {self.nowCH / max(1, self.nowRR)} {(self.CHR_target + CHR_tolerance)}")
        self.nowRR += 1

        #adjust theta based on "CHR_tolerance range vs CHR_target"
        nowCHR = self.nowCH / max(1, self.nowRR)
        if((self.CHR_target - CHR_tolerance) <= nowCHR <= (self.CHR_target + CHR_tolerance)):
            pass
        else:
            if(nowCHR < (self.CHR_target - CHR_tolerance)):
                print(f"theta {self.theta_base} -> {max(0, self.theta_base - 0.025)}") 
                self.theta_base = max(0.1, self.theta_base - 0.025)
            else:
                print(f"theta {self.theta_base} -> {min(1, self.theta_base + 0.025)}") 
                self.theta_base = min(0.975, self.theta_base + 0.025)

        #make query vector
        key_vec = prefer_emb.copy().reshape(1,-1)
        faiss.normalize_L2(key_vec) 

        if(self.vecDB.ntotal > 0):
            dist_list, index_list = self.vecDB.search(key_vec, min(self.vecDB.ntotal, 10))

            for similarity, index in zip(dist_list[0], index_list[0]):
                index2uid = self.index2id[index] #convert index -> uid
                similarity = round(similarity, 5)

                user_thresh = self.theta_base

                if(similarity >= user_thresh): 
                    self.nowCH += 1

                    print(f"{user_thresh} {similarity} {index} hit! ({self.nowCH / max(1, self.nowRR)})")
                    return similarity, self.userDB[index2uid].score, self.userDB[index2uid].relative_score, index2uid, index, 1

                else: 
                    print(f"{user_thresh} {similarity} {index} miss! ({self.nowCH / max(1, self.nowRR)})")
                    return similarity, None, None, None, None, 0
        else:
            print("empty!!")
            return 0, None, None, None, None, 0

    

    def cacheInsert_preferApprox(self, uid, item_emb, prefer_emb, score, relative_score, answer):     
        key_vec = prefer_emb.copy().reshape(1,-1)
        faiss.normalize_L2(key_vec) 
        
        before = self.vecDB.ntotal

        self.vecDB.add_with_ids(key_vec, self.index) 
        self.index2id[self.index] = uid 

        print(f"insert UID {uid}'s cache {self.index}")

        #meta update
        self.userDB[uid].score = score
        self.userDB[uid].relative_score = relative_score

        #meta arch update
        self.index += 1
    
    
    def cacheSize(self):
        vecDB_size = self.vecDB.ntotal * ((4 * self.input_dim) + (self.param * 2 * 4))

        resultDB_size = asizeof.asizeof(self.userDB)
        print(f"Cache size: {self.vecDB.ntotal} caches {vecDB_size + resultDB_size} bytes")
        return (vecDB_size + resultDB_size)
    

class GreenRecPlanner():
    def __init__(self, trace_op, HR_target, update_param):
        self.trace_op = trace_op

        self.HR_target = HR_target
        self.E_Look = 0.023468475  
        self.E_Inf = 0.310933862  

        self.window = 0
        self.Hit10_now = 0
        self.R_now = 0

        self.Error = -1
        self.update_param = update_param

        self.opt_theta = None
    
    # HIT10 및 CHR 함수 정의
    def hit10_function(self, theta_base):
        if(self.trace_op):
            return 0.4747 * (theta_base ** 2) -0.1632 * (theta_base) + 0.3885 #movieLens
        else:
            return 0.0707 * (theta_base ** 2) + 0.0982 * (theta_base) + 0.3315 #Twitch

    def chr_function(self, theta_base):
        if(self.trace_op):
            return min(max(0, -2.2057 * (theta_base ** 2) + 1.1655 * (theta_base) + 0.8977), 1) #movieLens
        else:
            return min(max(0, -1.1557 * (theta_base ** 2) -0.1407 * (theta_base) + 1.0581), 1) #Twitch

    
    def cost_function(self, theta_t_values, CI_t_values, R_t_values):
        CF = 0
        R_sum = self.R_now
        Hit10_sum = self.Hit10_now

        for t in range(self.window):
            theta_t = theta_t_values[t]
            R_t = R_t_values[t]
            CI_t = CI_t_values[t]

            Hit10_t = R_t * self.hit10_function(theta_t)
            CHR_t = self.chr_function(theta_t)

            Hit10_sum += Hit10_t
            R_sum += R_t

            CF += 0.000277778 * CI_t * R_t * (self.E_Look + (1 - CHR_t) * self.E_Inf)

        return CF  
    
    def constraint(self, theta_t_values, CI_t_values, R_t_values):
        R_sum = self.R_now
        Hit10_sum = self.Hit10_now

        for t in range(self.window):
            theta_t = theta_t_values[t]
            R_t = R_t_values[t]
            Hit10_t = R_t * self.hit10_function(theta_t)

            Hit10_sum += Hit10_t
            R_sum += R_t
        
        return (Hit10_sum / R_sum) - self.HR_target - 1e-6
    
    def NLPSolver(self, CI_t, R_t):

        if(self.Error > self.update_param or self.Error == -1):
            self.window = CI_t.size

            theta_0 = np.full(self.window, 0.5375)

            con = {'type': 'ineq', 'fun': lambda theta: self.constraint(theta, CI_t, R_t)}
            bounds = [(0.1, 0.975) for _ in range(self.window)] 

            result = minimize(self.cost_function, 
                              theta_0, 
                              args=(CI_t, R_t), 
                              method='SLSQP', 
                              bounds=bounds, 
                              constraints=[con], 
                              options={'disp': False})

            # 결과 확인
            if result.success:
                optimal_theta = result.x
                self.opt_theta = list(optimal_theta)
                print(f"Optimal control sequence: error {self.Error} opt_theta: {self.opt_theta}")
                self.Error = 0

                opt_choice = self.opt_theta.pop(0)
                print(f"opt_theta: {opt_choice} EXP_CHR: {self.chr_function(opt_choice)} EXP_HIT: {self.hit10_function(opt_choice)}")
                return 1, opt_choice, self.chr_function(opt_choice), self.hit10_function(opt_choice)
            else:
                print(f"Optimization failed: error {self.Error} opt_theta: {self.opt_theta}")

                if(self.opt_theta):
                    opt_choice = self.opt_theta.pop(0)
                    return 1, opt_choice, self.chr_function(opt_choice), self.hit10_function(opt_choice)
                else:
                    return 1, 0.7, self.chr_function(0.7), self.hit10_function(0.7) 
        else:
            print(f"Pass Optimiztion: error {self.Error} opt_theta: {self.opt_theta}")

            if(self.opt_theta):
                opt_choice = self.opt_theta.pop(0)
                return 0, opt_choice, self.chr_function(opt_choice), self.hit10_function(opt_choice)
            else:
                return 0, 0.7, self.chr_function(0.7), self.hit10_function(0.7)

    def updateState(self, Hit10_exp, Hit10_delta, R_delta):
        self.Hit10_now += Hit10_delta
        self.R_now += R_delta
        self.Error += abs(Hit10_exp - Hit10_delta)