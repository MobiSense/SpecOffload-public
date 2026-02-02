import math
import itertools



class HardwareProfile:
    def __init__(self, **kwargs):
        # --- 显存 ---
        self.GPU_MEM_CAPACITY_GB = kwargs.get('GPU_MEM_CAPACITY_GB', 24.0)
        

        self.v_target_prefill = kwargs.get('v_target_prefill', 4.0) 
        self.v_target_kv_prefill = kwargs.get('v_target_kv_prefill', 0.0001) 
        self.t_io_prefill = kwargs.get('t_io_prefill', 0.05) 
        self.t_target_comp_prefill = kwargs.get('t_target_comp_prefill', 0.01) #
        
        # 解码阶段 (Decoding)
        self.n_layer = kwargs.get('n_layer', 80) 
        self.v_target_ffn_decode = kwargs.get('v_target_ffn_decode', 2.0) 
        
        # 草稿模型 (Draft Model)
        self.p_correct_token = kwargs.get('p_correct_token', 0.7) 
        self.v_draft = kwargs.get('v_draft', 14.0) 
        self.v_draft_kv_decode = kwargs.get('v_draft_kv_decode', 0.00005) 
        self.t_draft_prefill_per_batch = kwargs.get('t_draft_prefill_per_batch', 0.1) 
        self.t_draft_decode_per_token = kwargs.get('t_draft_decode_per_token', 0.005) 
        
        # 目标模型解码 (Target Model Decoding)
        self.t_target_attention_cpu_per_item = kwargs.get('t_target_attention_cpu_per_item', 0.001) 
        self.t_target_ffn_io = kwargs.get('t_target_ffn_io', 0.002) 
        self.t_target_ffn_gpu = kwargs.get('t_target_ffn_gpu', 0.0001) 
        
class Policy:
    """存储规划器需要确定的参数策略"""
    def __init__(self, bs, bs_prefill, bs_draft, n_cand):
        self.bs = bs             # 总批次大小
        self.bs_prefill = bs_prefill # 预填充阶段的批次大小
        self.bs_draft = bs_draft     # 草稿模型的批次大小
        self.n_cand = n_cand         # 草稿模型生成的候补 token 数

class ModelInputs:
    """存储当前请求的输入参数"""
    def __init__(self, l_input, n_iter):
        self.l_input = l_input # 输入长度
        self.n_iter = n_iter   # 生成迭代次数（轮数）


def calculate_expected_tokens(policy: Policy, hw: HardwareProfile):
    p = hw.p_correct_token
    n_cand = policy.n_cand
    
    if p == 1.0:
        return n_cand + 1
    if p == 0.0:
        return 1.0
        
    try:
        term1 = n_cand * (p ** (n_cand + 2))
        term2 = (n_cand + 1) * (p ** (n_cand + 1))
        numerator = term1 - term2 + 1
        denominator = 1 - p
        expected_val = numerator / denominator
        return expected_val
    except OverflowError:
        return 1.0 # 发生溢出时回退

def calculate_generated_tokens(policy: Policy, inputs: ModelInputs, e_n_generated: float):
    return policy.bs * inputs.n_iter * e_n_generated

def calculate_prefill_latency(policy: Policy, inputs: ModelInputs, hw: HardwareProfile):

    t_per_prefill_batch = hw.t_io_prefill + hw.t_target_comp_prefill
    

    num_prefill_batches = math.ceil(policy.bs / policy.bs_prefill)
    t_prefill = num_prefill_batches * t_per_prefill_batch
    return t_prefill

def calculate_decoding_latency_per_round(policy: Policy, inputs: ModelInputs, hw: HardwareProfile):
    num_draft_batches = math.ceil(policy.bs / policy.bs_draft)
    t_draft_gpu = hw.t_draft_prefill_per_batch + (policy.n_cand - 1) * hw.t_draft_decode_per_token
    t_draft = num_draft_batches * t_draft_gpu

    t_attention_cpu = policy.bs * policy.n_cand * hw.t_target_attention_cpu_per_item
    
    t_ffn_io = hw.t_target_ffn_io
    t_ffn_gpu = hw.t_target_ffn_gpu
    
    t_target_decoding_layer = max(t_attention_cpu, t_ffn_io) + t_ffn_gpu
    t_target_decoding = hw.n_layer * t_target_decoding_layer
    
    t_decoding_round = max(t_target_decoding, t_draft)
    
    return t_decoding_round

def calculate_total_latency(t_prefill: float, t_decoding_round: float, inputs: ModelInputs):
    t_decoding_total = inputs.n_iter * t_decoding_round
    return t_prefill + t_decoding_total

def check_memory_constraints(policy: Policy, inputs: ModelInputs, hw: HardwareProfile, e_n_generated: float):
    v_prefill = hw.v_target_prefill + (policy.bs_prefill * inputs.l_input * hw.v_target_kv_prefill)
    
    if v_prefill > hw.GPU_MEM_CAPACITY_GB:
        return False, "Prefill memory exceeded"
        
    avg_tokens_in_draft_kv = inputs.l_input + e_n_generated
    v_draft_kv = policy.bs_draft * avg_tokens_in_draft_kv * hw.v_draft_kv_decode
    
    v_decoding = hw.v_target_ffn_decode + hw.v_draft + v_draft_kv
    
    if v_decoding > hw.GPU_MEM_CAPACITY_GB:
        return False, "Decoding memory exceeded"
        
    return True, "OK"


def find_optimal_policy(hw: HardwareProfile, inputs: ModelInputs, search_space: dict):

    print(f"--- 开始在 {hw.GPU_MEM_CAPACITY_GB}GB 显存上搜索最优策略 ---")
    print(f"输入: 长度={inputs.l_input}, 迭代={inputs.n_iter}\n")
    
    best_throughput = 0.0
    best_policy = None
    checked_policies = 0
    valid_policies = 0

    keys, values = zip(*search_space.items())
    policy_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    total_policies = len(policy_combinations)
    print(f"总共需检查 {total_policies} 种策略组合...")

    for policy_dict in policy_combinations:
        policy = Policy(**policy_dict)
        checked_policies += 1

        e_n_generated = calculate_expected_tokens(policy, hw)

        is_valid, reason = check_memory_constraints(policy, inputs, hw, e_n_generated)
        if not is_valid:
            # print(f"  [跳过] {policy_dict} - {reason}")
            continue
            
        valid_policies += 1
        
        t_prefill = calculate_prefill_latency(policy, inputs, hw)
        t_decoding_round = calculate_decoding_latency_per_round(policy, inputs, hw)
        t_total = calculate_total_latency(t_prefill, t_decoding_round, inputs)
        

        n_total_tokens = calculate_generated_tokens(policy, inputs, e_n_generated)
        

        throughput = n_total_tokens / t_total
        
        if throughput > best_throughput:
            best_throughput = throughput
            best_policy = policy_dict

    print(f"\n--- 搜索完成 ---")
    print(f"检查了 {checked_policies}/{total_policies} 种策略")
    print(f"其中 {valid_policies} 种策略满足显存约束")
    
    if best_policy:
        print("\n*** 找到的最优策略 ***")
        print(f"  策略: {best_policy}")
        print(f"  吞吐量: {best_throughput:.2f} tokens/s")
    else:
        print("\n未找到满足显存约束的有效策略。请检查硬件配置或搜索空间。")
        
    return best_policy, best_throughput



if __name__ == "__main__":
    

    MOCK_HARDWARE_PROFILE = HardwareProfile(
        GPU_MEM_CAPACITY_GB=24.0, 
        p_correct_token=0.6,      
        n_layer=64,               
        
        # 预填充
        v_target_prefill=3.0,
        v_target_kv_prefill=0.00012,
        t_io_prefill=0.06,
        t_target_comp_prefill=0.02,
        
        # 解码
        v_target_ffn_decode=1.5,
        t_target_attention_cpu_per_item=0.0005,
        t_target_ffn_io=0.003,
        t_target_ffn_gpu=0.0002,
        
        # 草稿
        v_draft=14.0, # 假设是一个 7B 草稿模型
        v_draft_kv_decode=0.00006,
        t_draft_prefill_per_batch=0.08,
        t_draft_decode_per_token=0.004
    )
    
    # 2. 定义输入
    MOCK_MODEL_INPUTS = ModelInputs(
        l_input=512,  
        n_iter=20     
    )
    

    POLICY_SEARCH_SPACE = {
        'bs': [20*x for x in range(1, 50)],                     # 总批次大小
        'bs_prefill': [20*x for x in range(1, 50)],            # 预填充批次
        'bs_draft': [4*x for x in range(1,5)],                 # 草稿模型批次
        'n_cand': [x for x in range(1, 5)]                     # 候补 token 数
    }
    
    import time
    start_time = time.time()
    find_optimal_policy(MOCK_HARDWARE_PROFILE, MOCK_MODEL_INPUTS, POLICY_SEARCH_SPACE)
    print(f"\n总搜索时间: {time.time() - start_time:.2f} 秒")