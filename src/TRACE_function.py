from collections import defaultdict

class recTrace:
    class energyConsumption:
        def __init__(self):
            self.PACKAGE_E = 0
            self.DRAM_E = 0
            self.GPU_E = 0
    class carbonFootprint:
        def __init__(self, ci):
            self.CI = ci
            self.PACKAGE_CF = 0
            self.DRAM_CF = 0
            self.GPU_CF = 0
    class request:
        def __init__(self, uid, flag, log, answer):
            self.UID = uid
            self.FLAG = flag
            self.LOG = log
            self.ANSWER = answer
    class user:
        def  __init__(self, uid, index):
            self.UID = uid
            self.index = index #user의 request item 위치

            self.history = []
            self.cacheHit = 0 #hit count
            self.cacheMiss = 0 #miss count
            self.hotness = 0

    def __init__(self, rq_start_time, rq_end_time, ci_start_time, ci_end_time):
        self.energyTrace = {}
        self.carbonTrace = {}
        self.requestTrace = {}
        self.historyTrace = defaultdict(list)
        self.targetTrace = defaultdict(list)
        self.userTrace = {}

        self.iteration = rq_end_time - rq_start_time + 1 #simulation length
        self.rq_start_time = rq_start_time
        self.rq_end_time = rq_end_time
        self.ci_start_time = ci_start_time
        self.ci_end_time = ci_end_time

        for time in range(rq_start_time, rq_end_time+1):
            self.requestTrace[time] = []
        self.requestTrace[0] = []

    
    def realworldRequestMonitor(self, RQfilename):
        n_ratings = 0
        min_time = 99999999999 
        max_time = 0

        #user 당 item list 추출
        file = f'../data/{RQfilename}.txt'

        for line in open(file, encoding='utf-8').readlines()[1:]:
            row = list(line.strip().split('\t'))

            UID = int(row[0])
            IID = int(row[1])
            TIME = int(row[2])

            self.historyTrace[UID].append(IID)

            if(self.rq_start_time <= TIME <= self.rq_end_time):
                self.targetTrace[UID].append(IID)
        
        #request trace 생성
        SEQ = 10

        for line in open(file, encoding='utf-8').readlines()[1:]:
            row = list(line.strip().split('\t'))

            UID = int(row[0])
            IID = int(row[1])
            TIME = int(row[2])

            if(self.rq_start_time <= TIME <= self.rq_end_time):
                min_time = min(min_time, TIME)
                max_time = max(max_time, TIME)

                #simulation 범위 내 user 수 추가
                if(UID not in self.userTrace.keys()):
                    if (len(self.historyTrace[UID]) > SEQ):
                        self.userTrace[UID] = self.user(UID, 0)
                    else:
                        continue
                
                #time에 해당하는 request 추가
                uid_index = self.userTrace[UID].index

                if(uid_index+SEQ < len(self.historyTrace[UID])-1):

                    log = self.historyTrace[UID][uid_index:uid_index+SEQ]
                    answer_index = min(uid_index+SEQ+9, len(self.historyTrace[UID]))
                    answer = self.historyTrace[UID][uid_index+SEQ:answer_index]
                    self.userTrace[UID].index += 1


                    self.requestTrace[TIME].append(self.request(UID, 0, log, answer))
                    n_ratings += 1

        print(f"logging RQ: {self.rq_start_time}~{self.rq_end_time}....")
        print(f"logging request: user {len(self.userTrace.keys())} => request {n_ratings}....") 
    

    def realworldCarbonMonitor(self, CIfilename):
        TIME = 1
        valid_index = 0

        file = f'../data/{CIfilename}.csv'

        for line in open(file, encoding='utf-8').readlines()[1:]:
            row = list(map(int, line.strip().split(',')))
            CI = row[1]
            
            if(self.ci_start_time <= TIME <= self.ci_end_time):
                self.energyTrace[self.rq_start_time + valid_index] = self.energyConsumption()
                self.carbonTrace[self.rq_start_time + valid_index] = self.carbonFootprint(CI)
                valid_index += 1

            TIME += 1

        print(f"logging CI: {self.ci_start_time}~{self.ci_end_time}...")



    def realworldMonitor(self, RQfilename, CIfilename):
        self.realworldRequestMonitor(RQfilename)
        self.realworldCarbonMonitor(CIfilename)


    #해당 시점에 요청된 한 user 수집
    def requestCall(self, timestamp):
        request_rate = 0
        requst_info = []

        for RQ in self.requestTrace[timestamp]: 
            requst_info.append((RQ.UID, RQ.LOG, RQ.ANSWER))
        
            RQ.FLAG = 1 #처리될 request이므로 flag on
            request_rate += 1

        return request_rate, requst_info


    def getFuture(self, nowTime):
        futureCI = []
        futureRR = []

        for time in range(nowTime, self.rq_end_time+1):
            RR = len(self.requestTrace[time])
            CI = self.carbonTrace[time].CI
            
            futureRR.append(RR)
            futureCI.append(CI)

        return futureRR, futureCI


    def carbonUpdate(self, timestamp, package_E, dram_E, gpu_E):
        self.energyTrace[timestamp].PACKAGE_E += package_E
        self.energyTrace[timestamp].DRAM_E += dram_E
        self.energyTrace[timestamp].GPU_E += gpu_E

        nowCI = self.carbonTrace[timestamp].CI
        package_CF = (package_E * nowCI * 0.00027778)
        dram_CF = (dram_E * nowCI * 0.00027778)
        gpu_CF = (gpu_E * nowCI * 0.00027778)

        self.carbonTrace[timestamp].PACKAGE_CF += package_CF
        self.carbonTrace[timestamp].DRAM_CF += dram_CF
        self.carbonTrace[timestamp].GPU_CF += gpu_CF

        return package_CF, dram_CF, gpu_CF

    def modelPrint(self, timestamp):
        energy_result = [self.energyTrace[timestamp].PACKAGE_E, self.energyTrace[timestamp].DRAM_E, self.energyTrace[timestamp].GPU_E]
        carbon_result = [self.carbonTrace[timestamp].PACKAGE_CF, self.carbonTrace[timestamp].DRAM_CF, self.carbonTrace[timestamp].GPU_CF]
        return energy_result, carbon_result