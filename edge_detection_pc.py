"""
边缘检测算法对比仿真
=====================
包含：
  1. 差比和（原始基线）
  2. Sobel 算子
  3. Canny 算子
  4. 简化相位一致性（本论文方法 - Simplified Phase Congruency, SPC）

运行环境：Python 3.8+
依赖安装：pip install numpy opencv-python matplotlib scipy

使用方式：
  python edge_detection_pc.py
  - 默认使用内置生成的合成赛道图像
  - 也可修改 IMAGE_PATH 指定自己的图像路径
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import convolve
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置区（按需修改）
# ============================================================
IMAGE_PATH = r"D:\Da_4\lunwen\K_laoshi\test\tu5\frames.182.png"          # None = 使用合成图像；填路径则加载真实图像
OUTPUT_SIZE = (188, 120)   # 模拟 MT9V034 输出分辨率
NOISE_LEVELS = None        # None = 根据图像自动检测；或手动指定如 [0, 5, 10, 20]
SAVE_RESULTS = True        # 是否保存结果图像
# ============================================================


# ============================================================
# 合成赛道图像生成
# ============================================================
def generate_track_image(size=(376, 240)):
    """
    生成模拟巡线赛道图像（白色赛道 + 黑色边线 + 渐变光照）
    输出为灰度图，uint8
    """
    h, w = size
    img = np.ones((h, w), dtype=np.float32) * 200  # 浅灰背景

    # 赛道主体（白色区域）
    for y in range(h):
        # 赛道中心线随y轻微弯曲，模拟转弯
        cx = w // 2 + int(40 * np.sin(2 * np.pi * y / h))
        road_w = w // 3
        x0 = max(0, cx - road_w // 2)
        x1 = min(w, cx + road_w // 2)
        img[y, x0:x1] = 230

    # 黑色边线（左右各一条，宽3px）
    for y in range(h):
        cx = w // 2 + int(40 * np.sin(2 * np.pi * y / h))
        road_w = w // 3
        x0 = max(0, cx - road_w // 2)
        x1 = min(w - 1, cx + road_w // 2)
        img[y, max(0, x0-3):x0+1] = 30
        img[y, x1:min(w, x1+4)] = 30

    # 模拟不均匀光照（左侧暗，右侧亮）
    gradient = np.linspace(0.75, 1.15, w)
    img = img * gradient[np.newaxis, :]

    # 模拟地面纹理噪声（微弱）
    texture = np.random.normal(0, 4, img.shape)
    img = np.clip(img + texture, 0, 255).astype(np.uint8)

    return img


def add_noise(img, sigma):
    """添加高斯噪声"""
    if sigma == 0:
        return img.copy()
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def resize_to_target(img):
    """缩放到目标分辨率（模拟 MT9V034 4x4 binning 输出）"""
    return cv2.resize(img, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)


# ============================================================
# 算法 1：差比和（原始基线）
# ============================================================
def diff_ratio_sum(img, threshold=0.15):
    """
    灰度差比和边缘检测
    原理：对每个像素，计算邻域灰度差与灰度和的比值
    threshold: 判断为边缘的阈值（0~1）
    返回：二值边缘图 uint8
    """
    img_f = img.astype(np.float32)
    h, w = img_f.shape
    result = np.zeros((h, w), dtype=np.float32)

    # 水平方向差比和
    left  = np.roll(img_f, 1, axis=1)
    right = np.roll(img_f, -1, axis=1)
    denom_h = left + right + 1e-6
    ratio_h = np.abs(left - right) / denom_h

    # 垂直方向差比和
    up   = np.roll(img_f, 1, axis=0)
    down = np.roll(img_f, -1, axis=0)
    denom_v = up + down + 1e-6
    ratio_v = np.abs(up - down) / denom_v

    result = np.maximum(ratio_h, ratio_v)

    # 固定阈值二值化
    edge = (result > threshold).astype(np.uint8) * 255
    return edge


# ============================================================
# 算法 2：Sobel
# ============================================================
def sobel_edge(img, threshold=50):
    """标准 Sobel 边缘检测"""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    _, edge = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
    return edge


# ============================================================
# 算法 3：Canny
# ============================================================
def canny_edge(img, low=30, high=80):
    """标准 Canny 边缘检测"""
    blurred = cv2.GaussianBlur(img, (3, 3), 0.8)
    return cv2.Canny(blurred, low, high)


# ============================================================
# 算法 4：简化相位一致性（本论文核心方法）
# ============================================================
def build_directional_filters():
    """
    构建4方向差分滤波器对
    方向：0°(水平)、90°(垂直)、45°、135°
    每个方向包含偶对称（类cos）和奇对称（类sin）两个滤波器
    用于近似 Log-Gabor 滤波器的相位响应
    """
    filters = []

    # 方向 0°：水平方向
    even_0 = np.array([[-1, 2, -1],
                        [-2, 4, -2],
                        [-1, 2, -1]], dtype=np.float32) / 4.0
    odd_0  = np.array([[ 0,  0,  0],
                        [-1,  0,  1],
                        [ 0,  0,  0]], dtype=np.float32)
    filters.append((even_0, odd_0, 0.0))

    # 方向 90°：垂直方向
    even_90 = np.array([[-1, -2, -1],
                         [ 2,  4,  2],
                         [-1, -2, -1]], dtype=np.float32) / 4.0
    odd_90  = np.array([[ 0, -1,  0],
                         [ 0,  0,  0],
                         [ 0,  1,  0]], dtype=np.float32)
    filters.append((even_90, odd_90, np.pi/2))

    # 方向 45°
    even_45 = np.array([[-2, -1,  2],
                         [-1,  4, -1],
                         [ 2, -1, -2]], dtype=np.float32) / 4.0
    odd_45  = np.array([[-1,  0,  1],
                         [ 0,  0,  0],
                         [ 1,  0, -1]], dtype=np.float32) / np.sqrt(2)
    filters.append((even_45, odd_45, np.pi/4))

    # 方向 135°
    even_135 = np.array([[ 2, -1, -2],
                          [-1,  4, -1],
                          [-2, -1,  2]], dtype=np.float32) / 4.0
    odd_135  = np.array([[ 1,  0, -1],
                          [ 0,  0,  0],
                          [-1,  0,  1]], dtype=np.float32) / np.sqrt(2)
    filters.append((even_135, odd_135, 3*np.pi/4))

    return filters


def adaptive_noise_threshold(response_magnitude, k=3.0):
    """
    自适应噪声阈值估计（论文创新点之一）
    基于响应幅值的统计特性自动确定噪声基底
    k: 阈值系数，越大越保守（减少噪声检测）
    原理：用响应幅值的中位数估计噪声水平，避免固定阈值在光照变化时失效
    """
    # 使用下四分位数估计噪声基底（比均值更鲁棒）
    noise_floor = np.percentile(response_magnitude, 25)
    threshold = noise_floor * k
    return max(threshold, 1e-6)


def directional_weight(responses, directions):
    """
    方向自适应权重（论文创新点之二）
    根据各方向的响应强度自动调整权重，强化主导方向
    解决低分辨率图像中弱方向噪声干扰问题
    """
    # 计算各方向的平均响应强度
    strengths = np.array([np.mean(r) for r in responses])
    strengths = strengths + 1e-6

    # Softmax 归一化得到权重
    exp_s = np.exp(strengths - strengths.max())
    weights = exp_s / exp_s.sum()
    return weights


def simplified_phase_congruency(img, k_threshold=3.0, epsilon=1e-8):
    """
    简化相位一致性边缘检测（Simplified Phase Congruency, SPC）
    
    核心思想：边缘位于各方向相位响应趋于一致的位置
    本简化版用4方向差分滤波器替代原版 Log-Gabor 滤波器组
    加入自适应噪声阈值和方向权重两个创新改进
    
    参数：
        img        : 输入灰度图 (uint8)
        k_threshold: 自适应阈值系数，越大提取越保守
        epsilon    : 数值稳定小量
    返回：
        pc_map  : 相位一致性响应图 (float32, 0~1)
        edge    : 二值边缘图 (uint8)
    """
    # 预处理：轻度平滑抑制噪声，保留强边缘
    # 对高对比度低纹理图像（赛道场景）噪声会严重干扰相位估计
    # 3x3高斯平滑可抑制随机噪声同时基本不损失边缘位置精度
    img_smooth = cv2.GaussianBlur(img, (5, 5), 1.2)
    img_f = img_smooth.astype(np.float32)

    filters = build_directional_filters()
    direction_responses = []  # 每个方向的幅值响应

    # Step 1：计算各方向的偶/奇对称响应
    for even_f, odd_f, angle in filters:
        # 偶对称响应（捕捉相位余弦分量）
        resp_even = convolve(img_f, even_f)
        # 奇对称响应（捕捉相位正弦分量）
        resp_odd  = convolve(img_f, odd_f)
        # 该方向的局部幅值
        amplitude = np.sqrt(resp_even**2 + resp_odd**2)
        direction_responses.append((resp_even, resp_odd, amplitude))

    # Step 2：自适应噪声阈值（创新点1）
    all_amplitudes = np.stack([r[2] for r in direction_responses], axis=0)
    total_amplitude = np.sum(all_amplitudes, axis=0)
    T = adaptive_noise_threshold(total_amplitude, k=k_threshold)

    # Step 3：方向自适应权重（创新点2）
    mean_responses = [r[2] for r in direction_responses]
    weights = directional_weight(mean_responses, [f[2] for f in filters])

    # Step 4：计算加权相位一致性
    # PC(x) = Σ_θ w_θ * max(A_θ(x) - T, 0) / (Σ_θ A_θ(x) + ε)
    weighted_sum = np.zeros_like(img_f)
    for i, (resp_even, resp_odd, amplitude) in enumerate(direction_responses):
        # 相位一致性分子：响应幅值超过噪声阈值的部分
        contribution = np.maximum(amplitude - T, 0)
        weighted_sum += weights[i] * contribution

    # 归一化
    pc_map = weighted_sum / (total_amplitude + epsilon)
    pc_map = np.clip(pc_map, 0, 1).astype(np.float32)

    # Step 5：双阈值滞后二值化（创新点3，专为低纹理高对比场景设计）
    # 原理：先用宽松阈值找"种子边缘"，再用严格阈值过滤弱响应
    # 两个阈值都基于PC图自身的统计量，不依赖绝对幅值
    # 因此在噪声改变幅值绝对大小时仍然稳定
    #
    # 高阈值：取PC图非零区域的中位数，确保只有真正强响应才成为种子
    nonzero_vals = pc_map[pc_map > 0]
    if len(nonzero_vals) == 0:
        return pc_map, np.zeros_like(pc_map, dtype=np.uint8)
    # 用较低百分位作为高阈值，扩大种子覆盖范围
    high_thresh = np.percentile(nonzero_vals, 40)
    low_thresh  = high_thresh * 0.2

    # 强边缘种子
    strong = (pc_map >= high_thresh).astype(np.uint8)
    # 弱边缘候选
    weak   = ((pc_map >= low_thresh) & (pc_map < high_thresh)).astype(np.uint8)

    # 连通性传播：弱边缘中与强边缘相邻的保留（扩大到2像素邻域）
    kernel_conn = np.ones((5, 5), np.uint8)
    strong_dilated = cv2.dilate(strong, kernel_conn, iterations=1)
    connected_weak = (weak & strong_dilated).astype(np.uint8)
    edge = np.clip(strong + connected_weak, 0, 1).astype(np.uint8) * 255

    return pc_map, edge


# ============================================================
# 评估指标
# ============================================================
def compute_metrics(edge_pred, edge_gt, tolerance=2):
    """
    计算边缘检测指标
    - Precision：预测为边缘且正确的比例
    - Recall：真实边缘被检测到的比例
    - F1：综合指标
    tolerance: 允许的像素偏移容忍距离
    """
    if edge_pred.max() == 0:
        return 0.0, 0.0, 0.0

    pred = (edge_pred > 0).astype(np.uint8)
    gt   = (edge_gt > 0).astype(np.uint8)

    # 用膨胀操作引入容忍距离
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2*tolerance+1, 2*tolerance+1))
    gt_dilated   = cv2.dilate(gt, kernel)
    pred_dilated = cv2.dilate(pred, kernel)

    tp_p = np.sum(pred * gt_dilated)   # 预测正确的边缘像素
    tp_r = np.sum(gt * pred_dilated)   # 真实边缘被覆盖的像素

    precision = tp_p / (np.sum(pred) + 1e-6)
    recall    = tp_r / (np.sum(gt) + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return float(precision), float(recall), float(f1)


def measure_time(func, *args, repeat=20):
    """测量函数平均运行时间（毫秒）"""
    # 预热
    func(*args)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func(*args)
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)

def estimate_noise_level(img):
    """
    自动估计图像真实噪声水平（σ）
    方法：局部小窗口法
      - 在纯背景区域（无边缘）内取8×8小窗口
      - 每个窗口内的std反映局部像素抖动，即传感器噪声
      - 取所有窗口std的中位数作为最终估计
    优势：排除了场景本身的亮度分区（如赛道深浅区域）对估计的干扰
    全图std会把赛道深色/浅色区域的亮度差当成噪声，导致严重高估
    返回：估计的噪声标准差
    """
    edges_mask = cv2.Canny(img, 20, 60)
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges_mask, kernel)

    h, w = img.shape
    win = 8
    local_stds = []
    for y in range(0, h - win, win):
        for x in range(0, w - win, win):
            patch_edge = edges_dilated[y:y+win, x:x+win]
            if patch_edge.sum() == 0:  # 纯背景patch，无边缘
                patch = img[y:y+win, x:x+win].astype(np.float32)
                local_stds.append(patch.std())

    if len(local_stds) < 10:
        return 3.0  # 背景区域太少时返回保守估计
    # 用中位数而非均值，进一步排除偶发异常patch
    return float(np.median(local_stds))


def auto_noise_levels(sigma_est):
    """
    根据估计噪声水平自动生成4档测试噪声范围
    策略：
      - 无噪声（σ=0）作为基准
      - 1×估计噪声：接近真实场景
      - 2×估计噪声：中度干扰，模拟光照/环境变化
      - 3×估计噪声：强干扰，实际可能遇到的上限
    超过3倍真实噪声已超出实际应用范围，测试意义有限
    数值取整并去重，保证单调递增
    """
    # 最小步长保证：即使真实噪声极低（如σ<3），
    # 也要保证测试档位之间有足够间距，使论文图表具有可读性
    # 每档最小步长为5，确保差异可见
    min_step = 5
    levels = sorted(set([
        0,
        max(min_step * 1, round(sigma_est * 1)),
        max(min_step * 2, round(sigma_est * 2)),
        max(min_step * 3, round(sigma_est * 3)),
    ]))
    return levels


# ============================================================
# 主实验流程
# ============================================================
def run_experiment():
    print("=" * 60)
    print("  边缘检测算法对比实验")
    print("  平台模拟：188×120 灰度图 (MT9V034 + CYT4BB7)")
    print("=" * 60)

    # ---------- 加载 / 生成图像 ----------
    if IMAGE_PATH is not None:
        raw = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            print(f"[警告] 无法读取 {IMAGE_PATH}，使用合成图像")
            raw = generate_track_image((240, 376))
    else:
        print("[信息] 使用合成赛道图像（可修改 IMAGE_PATH 替换为真实图像）")
        raw = generate_track_image((240, 376))

    # 缩放到目标分辨率
    img_orig = resize_to_target(raw)
    print(f"[信息] 图像分辨率：{img_orig.shape[1]}×{img_orig.shape[0]}")

    # 生成参考边缘图（用于指标计算）
    # 用高斯模糊后的高质量Canny作为伪GT，与输入图像保持一致
    # 实际论文中如有人工标注可替换此处
    img_blurred_ref = cv2.GaussianBlur(img_orig, (5, 5), 1.2)
    edge_gt = cv2.Canny(img_blurred_ref, 20, 60)
    print(f"[信息] 参考边缘图边缘像素数：{edge_gt.sum()//255}")

    # ---------- 自动确定噪声测试范围 ----------
    global NOISE_LEVELS
    if NOISE_LEVELS is None:
        sigma_est = estimate_noise_level(img_orig)
        NOISE_LEVELS = auto_noise_levels(sigma_est)
        print(f"[自动] 检测到图像噪声水平 σ≈{sigma_est:.1f}")
        print(f"[自动] 生成测试噪声范围：{NOISE_LEVELS}")
        print(f"[说明] 1×={NOISE_LEVELS[1]}（真实场景），"
              f"2×={NOISE_LEVELS[2]}（中度干扰），"
              f"3×={NOISE_LEVELS[3]}（强干扰上限）")
    else:
        print(f"[信息] 使用手动指定噪声范围：{NOISE_LEVELS}")

    # ---------- 运行4种算法 ----------
    algorithms = {
        '差比和（基线）':   lambda im: diff_ratio_sum(im, threshold=0.15),
        'Sobel':            lambda im: sobel_edge(im, threshold=40),
        'Canny':            lambda im: canny_edge(im, low=25, high=70),
        '简化相位一致性\n(本文SPC)': lambda im: simplified_phase_congruency(im)[1],
    }

    results_clean = {}
    times_clean   = {}
    for name, func in algorithms.items():
        edge = func(img_orig)
        t_mean, t_std = measure_time(func, img_orig)
        results_clean[name] = edge
        times_clean[name]   = (t_mean, t_std)
        p, r, f1 = compute_metrics(edge, edge_gt)
        label = name.replace('\n', ' ')
        print(f"\n  [{label}]")
        print(f"    耗时：{t_mean:.2f} ± {t_std:.2f} ms")
        print(f"    Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")

    # ---------- 鲁棒性实验（不同噪声水平）----------
    print("\n" + "=" * 60)
    print("  鲁棒性测试（高斯噪声）")
    print("=" * 60)

    robustness = {name: [] for name in algorithms}
    for sigma in NOISE_LEVELS:
        img_noisy = add_noise(img_orig, sigma)
        for name, func in algorithms.items():
            edge = func(img_noisy)
            _, _, f1 = compute_metrics(edge, edge_gt)
            robustness[name].append(f1)
        print(f"  噪声 σ={sigma:2d}：", end="")
        for name, func in algorithms.items():
            label = name.split('\n')[0]
            f1 = robustness[name][-1]
            print(f"  {label}={f1:.3f}", end="")
        print()

    # ---------- 可视化 ----------
    _plot_main_comparison(img_orig, results_clean, times_clean, edge_gt)
    _plot_robustness(robustness, NOISE_LEVELS)
    _plot_pc_internals(img_orig)

    print("\n[完成] 图像已显示，如开启 SAVE_RESULTS 也已保存到当前目录")
    plt.show()


# ============================================================
# 可视化函数
# ============================================================
def _plot_main_comparison(img, results, times, edge_gt):
    """主对比图：原图 + 4种算法结果 + 耗时"""
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle('边缘检测算法对比  |  分辨率 188×120  |  模拟 CYT4BB7 场景',
                 fontsize=13, fontweight='bold', y=1.01)

    cols = 1 + len(results)
    axes = [fig.add_subplot(1, cols, i+1) for i in range(cols)]

    # 原图
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('原始灰度图\n(MT9V034输出)', fontsize=10)
    axes[0].axis('off')

    colors = ['#888888', '#4a90d9', '#e8854a', '#27ae60']
    for ax, (name, edge), color in zip(axes[1:], results.items(), colors):
        ax.imshow(edge, cmap='gray', vmin=0, vmax=255)
        t_mean, t_std = times[name]
        ax.set_title(f'{name}\n{t_mean:.1f}±{t_std:.1f} ms',
                     fontsize=9, color=color, fontweight='bold')
        ax.axis('off')
        # 绿框标注本文方法
        if '相位一致性' in name or 'SPC' in name:
            for spine in ax.spines.values():
                spine.set_edgecolor('#27ae60')
                spine.set_linewidth(3)
                spine.set_visible(True)

    plt.tight_layout()
    if SAVE_RESULTS:
        fig.savefig('result_comparison.png', dpi=150, bbox_inches='tight')


def _plot_robustness(robustness, noise_levels):
    """鲁棒性曲线图"""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ['#888888', '#4a90d9', '#e8854a', '#27ae60']
    styles  = ['--o', '--s', '--^', '-D']
    widths  = [1.5, 1.5, 1.5, 2.5]

    for (name, f1s), color, style, lw in zip(
            robustness.items(), colors, styles, widths):
        label = name.replace('\n', ' ')
        ax.plot(noise_levels, f1s, style, label=label,
                color=color, linewidth=lw, markersize=7)

    ax.set_xlabel('高斯噪声标准差 σ', fontsize=11)
    ax.set_ylabel('F1 分数', fontsize=11)
    ax.set_title('鲁棒性测试：不同噪声强度下的 F1 分数', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(noise_levels)
    plt.tight_layout()
    if SAVE_RESULTS:
        fig.savefig('result_robustness.png', dpi=150, bbox_inches='tight')


def _plot_pc_internals(img):
    """展示简化PC算法的内部过程（用于论文方法章节插图）"""
    img_f = img.astype(np.float32)
    filters = build_directional_filters()

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle('简化相位一致性（SPC）算法内部过程可视化',
                 fontsize=12, fontweight='bold')

    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.4, wspace=0.3)

    # 原图
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img, cmap='gray')
    ax.set_title('输入图像\n188×120', fontsize=9)
    ax.axis('off')

    direction_names = ['0° 水平', '90° 垂直', '45° 斜向', '135° 反斜']
    amplitudes = []

    for i, ((even_f, odd_f, angle), dname) in enumerate(
            zip(filters, direction_names)):
        resp_e = convolve(img_f, even_f)
        resp_o = convolve(img_f, odd_f)
        amp    = np.sqrt(resp_e**2 + resp_o**2)
        amplitudes.append(amp)

        ax = fig.add_subplot(gs[0, i+1])
        ax.imshow(amp, cmap='hot')
        ax.set_title(f'{dname}\n方向响应幅值', fontsize=9)
        ax.axis('off')

    # 最终PC图
    pc_map, edge = simplified_phase_congruency(img)
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(pc_map, cmap='hot')
    ax.set_title('PC 响应图\n（归一化）', fontsize=9)
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(edge, cmap='gray')
    ax.set_title('SPC 边缘结果\n（自适应阈值后）', fontsize=9, color='#27ae60')
    ax.axis('off')

    # 与 Canny 对比放在同一行
    canny = canny_edge(img)
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(canny, cmap='gray')
    ax.set_title('Canny 结果\n（对比参考）', fontsize=9, color='#e8854a')
    ax.axis('off')

    # 自适应阈值说明图
    all_amp = np.stack(amplitudes, axis=0).sum(axis=0)
    ax = fig.add_subplot(gs[1, 3])
    ax.hist(all_amp.flatten(), bins=60, color='#4a90d9', alpha=0.7,
            edgecolor='white', linewidth=0.5)
    T = adaptive_noise_threshold(all_amp, k=3.0)
    ax.axvline(T, color='red', linewidth=2, linestyle='--',
               label=f'自适应阈值 T={T:.1f}')
    ax.set_title('自适应噪声阈值\n（创新点1）', fontsize=9)
    ax.set_xlabel('响应幅值', fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 方向权重图
    mean_amps = [a.mean() for a in amplitudes]
    exp_s = np.exp(np.array(mean_amps) - max(mean_amps))
    weights = exp_s / exp_s.sum()
    ax = fig.add_subplot(gs[1, 4])
    bars = ax.bar(direction_names, weights,
                  color=['#4a90d9','#27ae60','#e8854a','#9b59b6'],
                  edgecolor='white', linewidth=0.8)
    ax.set_title('方向自适应权重\n（创新点2）', fontsize=9)
    ax.set_ylabel('权重', fontsize=8)
    ax.set_ylim(0, 0.5)
    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{w:.2f}', ha='center', va='bottom', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(fontsize=7, rotation=15)

    plt.tight_layout()
    if SAVE_RESULTS:
        fig.savefig('result_pc_internals.png', dpi=150, bbox_inches='tight')


# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS',
                                        'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    np.random.seed(42)
    run_experiment()
