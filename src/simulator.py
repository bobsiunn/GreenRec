import TRACE_function, ENERGY_function, GreenRec_function, BERT4Rec_function
from tqdm import tqdm
import numpy as np
import random

TRACE_FILE_NAME = dict({0: 'movieLens_30min', 1: 'Twitch_30min'})
CI_FILE_NAME = dict({0: 'carbon_intensity_full'})
N_ITEMS = dict({0 : 3681, 1: 46875})

REQUEST_CNT = 0
CARBON_SUM = 0
OPTIMIZE_SUM = 0
SEARCH_SUM = 0
EXTRACT_SUM = 0
INSERT_SUM = 0
INFERENCE_SUM = 0
SAVING_SUM = 0

HIT10_SUM = 0
HIT20_SUM = 0
HIT50_SUM = 0
HIT100_SUM = 0

HIT_SUM = 0
MISS_SUM = 0


def init_Trace(args):
    trace = TRACE_function.recTrace(args.rq_start_time, args.rq_end_time, args.ci_start_time, args.ci_end_time)
    trace.realworldMonitor(TRACE_FILE_NAME[args.trace], CI_FILE_NAME[0])

    return trace

def init_Model(args):
    inference_model = BERT4Rec_function.load_model(args.trace)
    
    energy_model = ENERGY_function.energyModel()
    GreenRec_recommder = GreenRec_function.GreenRecRecomender(dim=64, param=16)
    GreenRec_planner = GreenRec_function.GreenRecPlanner(trace_op=args.trace, HR_target=args.const, update_param=args.delta) 

    return energy_model, inference_model, GreenRec_recommder, GreenRec_planner



def get_hit(trace_op, pred_score, item_seq, true_list):
    _all_items = set([i for i in range(1, N_ITEMS[trace_op] + 1)])
    nge_samples = random.sample(list(_all_items - set(item_seq)), 99)
    candidates = true_list + nge_samples

    pred_ranking = (-pred_score[candidates]).argsort().argsort()
    item_ranking = pred_ranking[0]

    topK = [10, 20, 50, 100]
    result = []

    for granularity in topK:
        if(item_ranking < granularity):
            result.append(1)
        else:
            result.append(0)
    return result


def call_optimize(trace, timestamp, mpc_model, energy_model):
    global OPTIMIZE_SUM

    R_t, CI_t = trace.getFuture(timestamp)
    optFlag, theta_base, expCHR, expHR = mpc_model.NLPSolver(CI_t=np.array(CI_t), R_t=np.array(R_t))

    if(optFlag):
        latency, package_E, dram_E, _ = energy_model.measure_virutal_ENERGY(0)
    else:
        latency, package_E, dram_E, _= 0, 0, 0, 0

    package_CF, dram_CF, _ = trace.carbonUpdate(timestamp, package_E, dram_E, 0)
    print(f"OPTIMIZE_E(LATENCY,PACK,DRAM,GPU): {latency:.9f} {package_E:.9f} {dram_E:.9f} {0}")
    print(f"OPTIMIZE_C(LATENCY,PACK,DRAM,GPU): {latency:.9f} {package_CF:.9f} {dram_CF:.9f} {0}")

    OPTIMIZE_SUM += (package_CF + dram_CF + 0)
    
    return theta_base, expCHR, expHR


def call_emb(trace, timestamp, item_seq, infer_model, energy_model):
    global INFERENCE_SUM

    item_emb, pos_emb, norm_emb = BERT4Rec_function.get_emb(item_seq, infer_model)
    item_emb = item_emb.cpu().tolist()[0]
    flat_emb = norm_emb.view(1,-1).cpu().detach().numpy()

    latency, package_E, dram_E, gpu_E = energy_model.measure_virutal_ENERGY(1)
    package_CF, dram_CF, gpu_CF = trace.carbonUpdate(timestamp, package_E, dram_E, gpu_E)

    print(f"EMBEDDING_E(LATENCY,PACK,DRAM,GPU): {latency:.9f} {package_E:.9f} {dram_E:.9f} {gpu_E:.9f}")
    print(f"EMBEDDING_C(LATENCY,PACK,DRAM,GPU): {latency:.9f} {package_CF:.9f} {dram_CF:.9f} {gpu_CF:.9f}") 

    INFERENCE_SUM += (package_CF + dram_CF + gpu_CF)
    
    return item_emb, pos_emb, norm_emb, flat_emb

def call_preferV(trace, timestamp, uid, item_emb, GreenRec_recommder, energy_model):
    global EXTRACT_SUM

    item_emb.pop()
    item_emb = np.array(item_emb).astype(np.float32)
    prefer_emb = GreenRec_recommder.predictPreference(uid, item_emb)

    latency, package_E, dram_E, gpu_E = energy_model.measure_virutal_ENERGY(2)
    package_CF, dram_CF, gpu_CF = trace.carbonUpdate(timestamp, package_E, dram_E, gpu_E)

    print(f"KALMAN_E(LATENCY,PACK,DRAM,GPU): {latency:.9f} {package_E:.9f} {dram_E:.9f} {gpu_E:.9f}")
    print(f"KALMAN_C(LATENCY,PACK,DRAM,GPU): {latency:.9f} {package_CF:.9f} {dram_CF:.9f} {gpu_CF:.9f}") 

    EXTRACT_SUM += (package_CF + dram_CF + 0)

    return prefer_emb

def call_searchV(trace, timestamp, uid, item_seq, query_vec, GreenRec_recommder, energy_model):
    global SEARCH_SUM

    _, score, relative_score, _, _, cacheHitFlag = GreenRec_recommder.cacheLookup_preferApprox(uid, item_seq, query_vec)

    latency, package_E, dram_E, _ = energy_model.measure_virutal_ENERGY(3)
    package_CF, dram_CF, _ = trace.carbonUpdate(timestamp, package_E, dram_E, 0)

    print(f"SEARCH_E(LATENCY,PACK,DRAM,GPU): {latency:.9f} {package_E:.9f} {dram_E:.9f} {0}")
    print(f"SEARCH_C(LATENCY,PACK,DRAM,GPU): {latency:.9f} {package_CF:.9f} {dram_CF:.9f} {0}")

    SEARCH_SUM += (package_CF + dram_CF + 0)

    return score, relative_score, cacheHitFlag

def call_insertV(trace, timestamp, uid, item_seq, prefer_vec, score, relative_score, answer, GreenRec_recommder, energy_model):
    global INSERT_SUM

    GreenRec_recommder.cacheInsert_preferApprox(uid=uid, 
                                            item_emb=item_seq,
                                            prefer_emb=prefer_vec,
                                            score=score,
                                            relative_score=relative_score,
                                            answer=answer)
    latency, package_E, dram_E, _ = energy_model.measure_virutal_ENERGY(4)
    package_CF, dram_CF, _ = trace.carbonUpdate(timestamp, package_E, dram_E, 0)

    print(f"INSERT_E(LATENCY,PACK,DRAM,GPU): {latency:.9f} {package_E:.9f} {dram_E:.9f} {0}")
    print(f"INSERT_C(LATENCY,PACK,DRAM,GPU): {latency:.9f} {package_CF:.9f} {dram_CF:.9f} {0}")

    INSERT_SUM += (package_CF + dram_CF + 0)

def call_hidden(trace, timestamp, input, infer_model, energy_model):
    global INFERENCE_SUM

    infer_result = BERT4Rec_function.inference(input, infer_model, K=20)

    latency, package_E, dram_E, gpu_E = energy_model.measure_virutal_ENERGY(5)
    package_CF, dram_CF, gpu_CF = trace.carbonUpdate(timestamp, package_E, dram_E, gpu_E)

    print(f"INFERENCE_E(LATENCY,PACK,DRAM,GPU): {latency:.9f} {package_E:.9f} {dram_E:.9f} {gpu_E:.9f}")
    print(f"INFERENCE_C(LATENCY,PACK,DRAM,GPU): {latency:.9f} {package_CF:.9f} {dram_CF:.9f} {gpu_CF:.9f}") 

    INFERENCE_SUM += (package_CF + dram_CF + gpu_CF)

    return infer_result

def call_cacheSave(trace, timestamp, GreenRec_recommder, energy_model):
    global SAVING_SUM

    cache_size = GreenRec_recommder.cacheSize() 
    _, _, dram_E, _ = energy_model.measure_virutal_ENERGY(6)

    memory_usage = cache_size / 67634200576 #usage(percent)
    memory_cost = memory_usage * dram_E #usage x static energy

    _, dram_CF, _ = trace.carbonUpdate(timestamp, 0, memory_cost, 0)
    print(f"SAVING_E(PACK,DRAM,GPU): {0} {memory_cost:.9f} {0}")
    print(f"SAVING_C(PACK,DRAM,GPU): {0} {dram_CF:.9f} {0}") 

    SAVING_SUM += (0 + dram_CF + 0)


def process(trace, timestamp, trace_op, mode, inference_model, GreenRec_recommder, energy_model):
    global REQUEST_CNT, HIT10_SUM, HIT20_SUM, HIT50_SUM, HIT100_SUM, HIT_SUM, MISS_SUM

    request_rate, request_info = trace.requestCall(timestamp) 
    num_hitK = 0

    if(request_rate):
        if(not mode):

            for uid, log, answer in request_info:
                GreenRec_recommder.metaRegister(uid)

                item_emb, _, norm_emb, _ = call_emb(trace=trace, 
                                                    timestamp=timestamp, 
                                                    item_seq=log,
                                                    infer_model=inference_model,
                                                    energy_model=energy_model)

                score, relative_score = call_hidden(trace=trace,
                                                   timestamp=timestamp,
                                                   input=(log, norm_emb),
                                                   infer_model=inference_model,
                                                   energy_model=energy_model)
                
                hitK = get_hit(trace_op=trace_op, pred_score=score, item_seq=log, true_list=answer[:1])

                num_hitK += hitK[0]
                HIT10_SUM += hitK[0]
                HIT20_SUM += hitK[1]
                HIT50_SUM += hitK[2]
                HIT100_SUM += hitK[3]
                REQUEST_CNT += 1
                MISS_SUM += 1

                print(f"UID: {uid} HIT@10: {hitK[0]} HIT@20: {hitK[1]} HIT@50: {hitK[2]} HIT@100: {hitK[3]}")
            
            return 0, num_hitK / request_rate, num_hitK

        else:
            hit_rate = 0
            miss_rate = 0

            for uid, log, answer in request_info:
                GreenRec_recommder.metaRegister(uid)

                item_emb, _, norm_emb, _ = call_emb(trace=trace, 
                                                    timestamp=timestamp, 
                                                    item_seq=log,
                                                    infer_model=inference_model,
                                                    energy_model=energy_model)

                prefer_emb = call_preferV(trace=trace, 
                                            timestamp=timestamp, 
                                            uid=uid, 
                                            item_emb=item_emb,  
                                            GreenRec_recommder=GreenRec_recommder, 
                                            energy_model=energy_model)
                
                score, relative_score, cacheHitFlag = call_searchV(trace=trace, 
                                                                    timestamp=timestamp, 
                                                                    uid=uid, 
                                                                    item_seq=log, 
                                                                    query_vec=prefer_emb, 
                                                                    GreenRec_recommder=GreenRec_recommder, 
                                                                    energy_model=energy_model)

                if(cacheHitFlag): #cache hit
                    hit_rate += 1

                    hitK = get_hit(trace_op=trace_op, pred_score=score, item_seq=log, true_list=answer[:1])

                    num_hitK += hitK[0]
                    HIT10_SUM += hitK[0]
                    HIT20_SUM += hitK[1]
                    HIT50_SUM += hitK[2]
                    HIT100_SUM += hitK[3]
                    REQUEST_CNT += 1
                    HIT_SUM += 1

                else: #cache miss
                    miss_rate += 1

                    score, relative_score = call_hidden(trace=trace,
                                                    timestamp=timestamp,
                                                    input=(log, norm_emb),
                                                    infer_model=inference_model,
                                                    energy_model=energy_model)
                    call_insertV(trace, timestamp, uid, log, prefer_emb, score, relative_score, answer, GreenRec_recommder, energy_model)

                    hitK = get_hit(trace_op=trace_op, pred_score=score, item_seq=log, true_list=answer[:1])
                    
                    num_hitK += hitK[0]
                    HIT10_SUM += hitK[0]
                    HIT20_SUM += hitK[1]
                    HIT50_SUM += hitK[2]
                    HIT100_SUM += hitK[3]
                    REQUEST_CNT += 1
                    MISS_SUM += 1

                print(f"UID: {uid} HIT@10: {hitK[0]} HIT@20: {hitK[1]} HIT@50: {hitK[2]} HIT@100: {hitK[3]}")
            return hit_rate / (hit_rate + miss_rate), num_hitK / request_rate, num_hitK
    else:
        return 0, 0, 0


def simulation(args):
    trace_op = args.trace
    mode = args.mode
    rq_start_time = args.rq_start_time
    rq_end_time = args.rq_end_time
    ci_start_time = args.ci_start_time
    ci_end_time = args.ci_end_time

    print(f"request trace: {trace_op} (0: MovieLens, 1: Twitch")
    print(f"simulation mode: {mode} (0: BASE, 1: GreenRec)")
    print(f"accuracy constraint: {args.const}")
    print(f"request simulation: {rq_start_time} ~ {rq_end_time}")
    print(f"carbon intensity simulation: {ci_start_time} ~ {ci_end_time}")
    print("==========================================\n\n")


    global CARBON_SUM, HIT_SUM, MISS_SUM, HIT10_SUM, HIT20_SUM, HIT50_SUM, HIT100_SUM, REQUEST_CNT

    trace = init_Trace(args)
    energy_model, inference_model, GreenRec_recommder, GreenRec_planner = init_Model(args)

    if(mode):
        print("GreenRec simulation")

        for timestamp in tqdm(range(rq_start_time,1+rq_end_time)):
            nowRR = len(trace.requestTrace[timestamp])
            nowCI = trace.carbonTrace[timestamp].CI
            print(f"TIME: {timestamp} RR: {nowRR} CI: {nowCI}")


            theta_base, expCHR, expHR = call_optimize(trace, timestamp, GreenRec_planner, energy_model)
            GreenRec_recommder.cacheTargetChange(new_CHR=expCHR, new_HR=expHR, nowRR=nowRR, baseTheta=theta_base)
            
            hit_rate, hitK, click = process(trace, timestamp, trace_op, mode, inference_model, GreenRec_recommder, energy_model)
            GreenRec_planner.updateState(expHR * nowRR, click, nowRR)
            print(f"Result: Actual CHR {hit_rate} Target CHR {expCHR} | Actual Accuracy {hitK} Target Accuracy {expHR}")
            
            call_cacheSave(trace, timestamp, GreenRec_recommder, energy_model)
            
            energy_result, carbon_result = trace.modelPrint(timestamp)
            print(f"TOTAL_E(PACK,DRAM,GPU): {energy_result[0]:.9f} {energy_result[1]:.9f} {energy_result[2]:.9f} {HIT10_SUM / max(1,REQUEST_CNT)} {hit_rate}")
            print(f"TOTAL_C(PACK,DRAM,GPU): {carbon_result[0]:.9f} {carbon_result[1]:.9f} {carbon_result[2]:.9f} {HIT10_SUM / max(1,REQUEST_CNT)} {hit_rate}") 

            CARBON_SUM += (carbon_result[0] + carbon_result[1] + carbon_result[2])
        
        print(f"CARBON: {CARBON_SUM} HIT_RATE: {HIT_SUM / (HIT_SUM + MISS_SUM)} AVG_HIT10: {HIT10_SUM / REQUEST_CNT} AVG_HIT20: {HIT20_SUM / REQUEST_CNT} AVG_HIT50: {HIT50_SUM / REQUEST_CNT} AVG_HIT100: {HIT100_SUM / REQUEST_CNT}")


    else:
        print("Baseline simulation")

        for timestamp in tqdm(range(rq_start_time,1+rq_end_time)):
            nowRR = len(trace.requestTrace[timestamp])
            nowCI = trace.carbonTrace[timestamp].CI
            print(f"TIME: {timestamp} => RR: {nowRR} CI: {nowCI}")

            hit_rate, hitK, click = process(trace, timestamp, trace_op, mode, inference_model, GreenRec_recommder, energy_model)
            print(f"Result: Actual CHR {hit_rate} | Actual Accuracy {hitK}")

            energy_result, carbon_result = trace.modelPrint(timestamp)
            print(f"TOTAL_E(PACK,DRAM,GPU): {energy_result[0]:.9f} {energy_result[1]:.9f} {energy_result[2]:.9f}")
            print(f"TOTAL_C(PACK,DRAM,GPU): {carbon_result[0]:.9f} {carbon_result[1]:.9f} {carbon_result[2]:.9f}") 

            CARBON_SUM += (carbon_result[0] + carbon_result[1] + carbon_result[2]) 

        print(f"CARBON: {CARBON_SUM} HIT_RATE: {HIT_SUM / (HIT_SUM + MISS_SUM)} AVG_HIT10: {HIT10_SUM / REQUEST_CNT} AVG_HIT20: {HIT20_SUM / REQUEST_CNT} AVG_HIT50: {HIT50_SUM / REQUEST_CNT} AVG_HIT100: {HIT100_SUM / REQUEST_CNT}")
    
    print("\n\n================== simulation result ====================")
    print(f"TOTAL CARBON: {CARBON_SUM}")
    print(f"OPTIMIZE CARBON: {OPTIMIZE_SUM}")
    print(f"SEARCH CARBON: {SEARCH_SUM}")
    print(f"EXTRACT CARBON: {EXTRACT_SUM}")
    print(f"INSERT CARBON: {INSERT_SUM}")
    print(f"SAVE CARBON: {SAVING_SUM}")
    print(f"INFERENCE CARBON: {INFERENCE_SUM}\n\n")

    print(f"CACHE HIT RATE: {HIT_SUM / (HIT_SUM + MISS_SUM)}")
    print(f"HIT10 : {HIT10_SUM / REQUEST_CNT}")
    print(f"Accuracy Constraint Guarantee: {(HIT10_SUM / REQUEST_CNT) / args.const}")