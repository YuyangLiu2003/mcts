import math
import random
import matplotlib.pyplot as plt
import numpy as np

def generate_random_comparisons(num_players, num_comparisons=None):
    """Generate random comparison data"""
    if num_comparisons is None:
        pairs = []
        for i in range(num_players):
            for j in range(i+1, num_players):
                pairs.append((i, j, random.randint(-5, 5)))
        return pairs
    else:
        pairs = []
        possible_pairs = [(i, j) for i in range(num_players) for j in range(num_players) if i < j]
        for _ in range(num_comparisons):
            i, j = random.choice(possible_pairs)
            pairs.append((i, j, random.randint(-5, 5)))
        return pairs


def compute_r_values(pairs_data, num_players, k=0.8, iter_mean=6, final_scale=3.8, score_range=(0,10)):
    """Compute ln(r) values with given comparison data using numpy for efficiency"""
    r = np.ones(num_players, dtype=np.float64)
    min_r = 1e-8
    
    # 初始化对手列表（保持列表形式，因每个玩家对手数量不同）
    opponents = [[] for _ in range(num_players)]
    sum_p = np.zeros(num_players, dtype=np.float64)
    
    # 处理配对数据，构建对手列表和sum_p
    for i, j, score in pairs_data:
        # k控制了pij对于score的敏感度，k越大，pij对score的变化越敏感
        # b控制了pij的基准值，避免r值迭代中太发散。b越大，最终r的分布越窄
        # p_ij = k * (score + 5) / 10 + b  # 对于有输赢分数的情况，用i对j胜利的概率来代替标准bt模型中的“战胜次数”，两者成正比
        p_ij = k * score / 10 + 0.5  # 对于有输赢分数的情况，用i对j胜利的概率来代替标准bt模型中的“战胜次数”，两者成正比
        p_ji = 1.0 - p_ij  # 计算j对i的胜利概率，构成了两条边，也形成了强连通网络
        
        opponents[i].append(j)
        opponents[j].append(i)
        sum_p[i] += p_ij
        sum_p[j] += p_ji
    
    max_iterations = 1000
    epsilon = 1e-8
    
    for iteration in range(max_iterations):
        r_old = r.copy()  # 保存当前r值用于后续比较
        
        # 逐个更新每个玩家的r值
        for i in range(num_players):
            js = opponents[i]
            if not js:  # 没有对手时，分母为0
                denominator = 0.0
            else:
                # 注意：r[i]是当前玩家i的r值，r[js]是所有对手j的r值（js是i的对手列表）
                denominator = np.sum(1.0 / (r[i] + r[js]))
            
            # 更新r值（满足条件时）
            if denominator > 0 and sum_p[i] > 0:
                new_r = sum_p[i] / denominator  # 迭代公式本质上和Zermelo算法一致，pij是似然函数的指数
                r[i] = max(new_r, min_r)
        
        # 在过程中把r的均值限制在iter_mean，避免算法发散。这不会改变r的比值
        r_sum = r.sum()
        if r_sum > 0:
            scale_factor = iter_mean * num_players / r_sum
            r *= scale_factor
            # 确保不小于最小阈值
            r = np.maximum(r, min_r)
        
        # 计算收敛差异（欧氏距离）
        diff = np.linalg.norm(r - r_old)
        if diff < epsilon:
            break
    
    # 将r值转化为自然对数值，因为BT模型在乎的是r值之间的比值
    bt_score = np.log(r)

    # 可选，最终将r放缩到需要的范围
    final_score = bt_score * final_scale
    min_val, max_val = score_range
    final_score = np.clip(final_score, min_val, max_val)

    return final_score

def analyze_kb_range(num_players=4, num_samples=1000, comparison_type="random", x_range=(-10, 13)):
    """Analyze impact of different k and b values on r-value distribution, generate matrix of subplots.
    
    Parameters:
        x_range (tuple): (min, max) for x-axis range. Bins will be generated with 0.1 step within this range.
    """
    print(f"Starting analysis of k and b parameter range impact on r-value distribution for {num_players} players...")
    print(f"Number of samples: {num_samples}")
    print(f"x-axis range: {x_range}")

    # Define ranges for k and b
    k_values = [0.6, 0.64, 0.66, 0.7]
    b_values = [0.1, 0.2, 0.3]
    
    # Create matrix of subplots (rows for b values x columns for k values)
    fig, axes = plt.subplots(len(b_values), len(k_values), figsize=(4*len(k_values), 4*len(b_values)))
    fig.suptitle(f'Impact of different k and b values on r-value distribution ({num_players} players)', 
                 fontsize=16, y=1.02)
    
    # Generate fixed comparison data for all parameter combinations to ensure fair comparison
    all_pairs_data = []
    for _ in range(num_samples):
        if comparison_type == "all":
            pairs = generate_random_comparisons(num_players)
        else:
            max_pairs = num_players * (num_players - 1) // 2
            min_pairs = num_players - 1
            num_comparisons = random.randint(min_pairs, max_pairs)
            pairs = generate_random_comparisons(num_players, num_comparisons)
        all_pairs_data.append(pairs)
    
    # Unpack x_range
    x_min, x_max = x_range
    bins = np.arange(x_min, x_max + 0.1, 0.1)  # ensure inclusive of x_max
    
    # Iterate through all k and b combinations
    for b_idx, b in enumerate(b_values):
        for k_idx, k in enumerate(k_values):
            # Get current axis
            if len(b_values) == 1 and len(k_values) == 1:
                ax = axes
            elif len(b_values) == 1:
                ax = axes[k_idx]
            elif len(k_values) == 1:
                ax = axes[b_idx]
            else:
                ax = axes[b_idx, k_idx]
            
            # Calculate all r values for current parameter combination
            all_r_values = []
            for pairs in all_pairs_data:
                r_values = compute_r_values(pairs, num_players, k)
                all_r_values.extend(r_values)
            
            # Plot histogram with custom x-range and 0.1 bin size
            ax.hist(all_r_values, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Set subplot properties
            ax.set_title(f'k={k}, b={b}', fontsize=12)
            ax.set_xlabel('BT score', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_xlim(x_min, x_max)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', labelsize=8)
            
            # Set x-ticks every 1.0 unit within the range
            tick_step = 1.0
            x_ticks = np.arange(
                np.floor(x_min / tick_step) * tick_step,
                np.ceil(x_max / tick_step) * tick_step + tick_step,
                tick_step
            )
            ax.set_xticks(x_ticks)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_path = "kb_range_3.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")

if __name__ == "__main__":
    analyze_kb_range(num_players=3, num_samples=2000, comparison_type="random")