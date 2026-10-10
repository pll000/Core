import re
from typing import List, Tuple, Optional, Dict
import numpy as np
from collections import defaultdict
from swift.plugin import ORM, orms


def answer_judge(response_letters: str, solution: str) -> Tuple[float, int]:
    """
    小题评分函数：对选项进行逐位比对,计算准确率(缺失也参与扣分)。
    返回 (准确率 [0.0~1.0], 正确题目个数)。
    """
    pred = ''.join(filter(str.isalpha, response_letters.upper()))
    truth = ''.join(filter(str.isalpha, solution.upper()))

    max_len = len(truth)
    if max_len == 0:
        return 0.0, 0

    score = 0.0
    correct_count = 0
    for i in range(max_len):
        pred_char = pred[i] if i < len(pred) else None
        truth_char = truth[i] if i < len(truth) else None
        if pred_char is not None and truth_char is not None and pred_char == truth_char:
            score += 1.0
            correct_count += 1

    final_score = score / max_len
    return final_score, correct_count


def ensure_group_solution_consistency_by_gt(group, rewards, g_idx) -> bool:
    """基于 gt_letters 的严格一致性校验(仅对 valid 样本判定)。"""
    gt_set = {r.get('gt_letters', '') for r in group if r.get('valid', False)}
    if len(gt_set) > 1:
        for r in group:
            rewards[r['idx']] = 0.0
        print(f"[Warning] Inconsistent ground-truth in group {g_idx}: {gt_set}. Set rewards to 0.")
        return False
    return True


def extract_process_length(comp: str) -> int:
    """提取过程长度 L。"""
    think_span = re.search(r'<think>(.*?)</think>', comp, re.IGNORECASE | re.DOTALL)
    if think_span:
        proc_text = think_span.group(1)
    else:
        proc_text = re.sub(r'<answer>.*?</answer>', '', comp, flags=re.IGNORECASE | re.DOTALL)

    proc_text = proc_text.strip()
    L = len(proc_text)
    return max(1, L)


def extract_think_text(comp: str) -> str:
    """提取 <think> 中的推理文本; 若不存在则返回去掉 <answer> 后的文本。"""
    think_span = re.search(r'<think>(.*?)</think>', comp, re.IGNORECASE | re.DOTALL)
    if think_span:
        return think_span.group(1).strip()

    return re.sub(r'<answer>.*?</answer>', '', comp, flags=re.IGNORECASE | re.DOTALL).strip()


# ==================== Counting 任务相关函数 ====================
def parse_counting_question_options(question_text: str) -> List[List[Tuple[str, str, float, float]]]:
    """从问题文本中解析三个问题的选项数据(计数任务)"""
    question_pattern = r'Please select the range.*?from Figure \d+ to Figure \d+.*?(?=Please select|Please answer|$)'
    questions = re.findall(question_pattern, question_text, re.IGNORECASE | re.DOTALL)

    if len(questions) < 3:
        parts = question_text.split('Please select the range')
        questions = [p for p in parts if 'Figure' in p][:3]

    all_options = []

    for q_text in questions[:3]:
        options = []
        option_pattern = r'([A-D]):\s*([+-])(\d+)\s+to\s+([+-])(\d+)'
        matches = re.findall(option_pattern, q_text, re.IGNORECASE)

        for match in matches:
            option_letter = match[0].upper()
            sign1 = match[1]
            low = float(match[2])
            sign2 = match[3]
            high = float(match[4])

            if sign1 != sign2:
                print(f"[Warning] Inconsistent signs in option {option_letter}: {sign1} vs {sign2}")
                continue

            options.append((option_letter, sign1, low, high))

        options.sort(key=lambda x: x[0])

        if len(options) == 4:
            all_options.append(options)
        else:
            print(f"[Warning] Expected 4 options, got {len(options)}. Text: {q_text[:100]}...")
            all_options.append([
                ('A', '+', 0, 100),
                ('B', '+', 100, 200),
                ('C', '+', 200, 300),
                ('D', '+', 300, 400)
            ])

    while len(all_options) < 3:
        all_options.append([
            ('A', '+', 0, 100),
            ('B', '+', 100, 200),
            ('C', '+', 200, 300),
            ('D', '+', 300, 400)
        ])

    return all_options[:3]


def check_counting_transitivity(ans: str, question_text: Optional[str] = None) -> int:
    """计数任务的传递性一致性检查"""
    ans = ans.strip().upper()

    if len(ans) != 3 or not all(c in 'ABCD' for c in ans):
        return 0

    if not question_text:
        return 0

    try:
        all_options = parse_counting_question_options(question_text)

        if len(all_options) != 3:
            return 0

        def get_option_by_letter(options_list, letter):
            for opt in options_list:
                if opt[0] == letter:
                    return opt
            return None

        opt_1_2 = get_option_by_letter(all_options[0], ans[0])
        opt_1_3 = get_option_by_letter(all_options[1], ans[1])
        opt_2_3 = get_option_by_letter(all_options[2], ans[2])

        if not all([opt_1_2, opt_1_3, opt_2_3]):
            return 0

        _, sign_1_2, low_1_2, high_1_2 = opt_1_2
        _, sign_1_3, low_1_3, high_1_3 = opt_1_3
        _, sign_2_3, low_2_3, high_2_3 = opt_2_3

        def to_signed_range(sign, low, high):
            if sign == '-':
                return (-high, -low)
            else:
                return (low, high)

        r1_2 = to_signed_range(sign_1_2, low_1_2, high_1_2)
        r1_3 = to_signed_range(sign_1_3, low_1_3, high_1_3)
        r2_3 = to_signed_range(sign_2_3, low_2_3, high_2_3)

        sum_low = r1_2[0] + r2_3[0]
        sum_high = r1_2[1] + r2_3[1]

        range_size_1_3 = abs(r1_3[1] - r1_3[0])
        range_size_sum = abs(sum_high - sum_low)

        avg_range = (range_size_1_3 + range_size_sum) / 2
        tolerance = max(avg_range * 0.5, 50)

        overlap_low = max(sum_low, r1_3[0])
        overlap_high = min(sum_high, r1_3[1])
        has_overlap = overlap_low <= overlap_high + tolerance

        center_sum = (sum_low + sum_high) / 2
        center_1_3 = (r1_3[0] + r1_3[1]) / 2
        center_distance = abs(center_sum - center_1_3)
        center_close = center_distance <= tolerance

        def get_trend(sign):
            return 1 if sign == '+' else -1

        trend_1_2 = get_trend(sign_1_2)
        trend_2_3 = get_trend(sign_2_3)
        trend_1_3 = get_trend(sign_1_3)

        sign_consistent = True

        if trend_1_2 == trend_2_3:
            if trend_1_2 != trend_1_3:
                sign_consistent = False

        if (has_overlap or center_close) and sign_consistent:
            return +1
        elif has_overlap or center_close:
            return 0
        else:
            return -1

    except Exception as e:
        print(f"[Warning] Counting transitivity check error: {e}")
        import traceback
        traceback.print_exc()
        return 0


# ==================== Depth 任务相关函数 ====================
def parse_depth_question_options(question_text: str) -> List[dict]:
    """从问题文本中解析每个子问题的选项映射(深度任务)"""
    question_pattern = r'Question\s+\d+:(.*?)(?=Question\s+\d+:|Please answer)'
    questions = re.findall(question_pattern, question_text, re.IGNORECASE | re.DOTALL)

    if len(questions) < 3:
        parts = question_text.split('Question')
        questions = [p for p in parts if p.strip() and any(c in p for c in 'ABC')]

    mappings = []

    for q_text in questions[:3]:
        mapping = {}
        option_pattern = r'([ABC])\s*:\s*(Decreases?|Increases?|No\s+change)'
        matches = re.findall(option_pattern, q_text, re.IGNORECASE)

        for option, change_type in matches:
            option = option.upper()
            change_type_lower = change_type.lower().replace(' ', '')

            if 'decrease' in change_type_lower:
                mapping[option] = -1
            elif 'increase' in change_type_lower:
                mapping[option] = +1
            else:
                mapping[option] = 0

        if len(mapping) == 3:
            mappings.append(mapping)
        else:
            print(f"[Warning] Failed to parse depth options, using default mapping. Text: {q_text[:100]}...")
            mappings.append({'A': -1, 'B': +1, 'C': 0})

    while len(mappings) < 3:
        mappings.append({'A': -1, 'B': +1, 'C': 0})

    return mappings[:3]


def check_depth_transitivity(ans: str, question_text: Optional[str] = None) -> int:
    """深度任务的传递性一致性检查"""
    ans = ans.strip().upper()

    if len(ans) != 3 or not all(c in 'ABC' for c in ans):
        return 0

    try:
        if question_text:
            mappings = parse_depth_question_options(question_text)
        else:
            default_mapping = {'A': -1, 'B': +1, 'C': 0}
            mappings = [default_mapping] * 3

        change_1_2 = mappings[0].get(ans[0], 0)
        change_1_3 = mappings[1].get(ans[1], 0)
        change_2_3 = mappings[2].get(ans[2], 0)

        expected_1_3 = change_1_2 + change_2_3

        if expected_1_3 == change_1_3:
            return +1
        else:
            if abs(expected_1_3 - change_1_3) <= 1:
                return 0
            else:
                return -1

    except Exception as e:
        print(f"[Warning] Depth transitivity check error: {e}")
        return 0


# ==================== Distance 任务相关函数 ====================
def parse_distance_question_options(question_text: str) -> List[dict]:
    """从问题文本中解析每个子问题的选项映射(距离任务)"""
    question_pattern = r'How does the distance.*?from Figure \d+ to Figure \d+\?(.*?)(?=How does the distance|Please answer|$)'
    questions = re.findall(question_pattern, question_text, re.IGNORECASE | re.DOTALL)

    if len(questions) < 3:
        parts = question_text.split('\n')
        questions = [p for p in parts if 'How does the distance' in p][:3]

    mappings = []

    for q_text in questions[:3]:
        mapping = {}
        option_pattern = r'([ABC])\s*:\s*(No\s+change|Increases?|Decreases?)'
        matches = re.findall(option_pattern, q_text, re.IGNORECASE)

        for option, change_type in matches:
            option = option.upper()
            change_type_lower = change_type.lower().replace(' ', '')

            if 'decrease' in change_type_lower:
                mapping[option] = -1
            elif 'increase' in change_type_lower:
                mapping[option] = +1
            else:
                mapping[option] = 0

        if len(mapping) == 3:
            mappings.append(mapping)
        else:
            print(f"[Warning] Failed to parse distance options, using default mapping. Text: {q_text[:100]}...")
            mappings.append({'A': -1, 'B': +1, 'C': 0})

    while len(mappings) < 3:
        mappings.append({'A': -1, 'B': +1, 'C': 0})

    return mappings[:3]


def check_distance_transitivity(ans: str, question_text: Optional[str] = None) -> int:
    """距离任务的传递性一致性检查"""
    ans = ans.strip().upper()

    if len(ans) != 3 or not all(c in 'ABC' for c in ans):
        return 0

    try:
        if question_text:
            mappings = parse_distance_question_options(question_text)
        else:
            default_mapping = {'A': -1, 'B': +1, 'C': 0}
            mappings = [default_mapping] * 3

        change_1_2 = mappings[0].get(ans[0], 0)
        change_1_3 = mappings[1].get(ans[1], 0)
        change_2_3 = mappings[2].get(ans[2], 0)

        expected_1_3 = change_1_2 + change_2_3

        if expected_1_3 == change_1_3:
            return +1
        else:
            if abs(expected_1_3 - change_1_3) <= 1:
                return 0
            else:
                return -1

    except Exception as e:
        print(f"[Warning] Distance transitivity check error: {e}")
        import traceback
        traceback.print_exc()
        return 0


# ==================== Relation 任务相关函数 ====================
def check_spatial_consistency(ans: str, question_text: Optional[str] = None) -> int:
    """空间关系的一致性检查(保守策略:不进行复杂逻辑验证)"""
    ans = ans.strip().upper()

    if len(ans) != 3 or not all(c in 'ABCD' for c in ans):
        return 0

    return 0


# ==================== Answer Mapping 相关函数 ====================
def get_counting_option_letter_from_value(diff_value: float,
                                          options_list: List[Tuple[str, str, float, float]]) -> Optional[str]:
    """根据有符号差值 diff_value 映射到 counting 题某一小题的选项字母。"""
    for option_letter, sign, low, high in options_list:
        if sign == '+':
            if low <= diff_value <= high:
                return option_letter
        elif sign == '-':
            if -high <= diff_value <= -low:
                return option_letter
    return None


# ==================== Counting 估计抽取：更鲁棒版本 ====================
def _normalize_count_value(v1: str, v2: Optional[str] = None) -> Optional[float]:
    """
    将单值或区间值转成 float
    - "350" -> 350
    - "300","400" -> 350
    """
    try:
        if v2 is None:
            return float(v1)
        a = float(v1)
        b = float(v2)
        return (a + b) / 2.0
    except Exception:
        return None


def _figure_alias_patterns():
    """
    为 1/2/3 三张图构造 alias pattern
    """
    return {
        1: r'(?:figure|fig|image)\s*1\b|(?:the\s+)?first\s+(?:image|figure)\b|(?:^|\W)f1(?:\W|$)',
        2: r'(?:figure|fig|image)\s*2\b|(?:the\s+)?second\s+(?:image|figure)\b|(?:^|\W)f2(?:\W|$)',
        3: r'(?:figure|fig|image)\s*3\b|(?:the\s+)?third\s+(?:image|figure)\b|(?:^|\W)f3(?:\W|$)',
    }


def _remove_option_like_text(text: str) -> str:
    """
    尽量移除/弱化题目选项、option解释等干扰内容，减少把范围选项数字误识别为人数估计。
    """
    lines = text.splitlines()
    kept = []

    drop_patterns = [
        r'\boptions?\b',
        r'\boption\s*[abcd]\b',
        r'\b[A-D]\s*:\s*[+-]?\d+\s*(?:to|-)\s*[+-]?\d+\b',
        r'\bfirst question\b',
        r'\bsecond question\b',
        r'\bthird question\b',
        r'\bthe options for\b',
        r'\blooking at the options\b',
        r'\bcheck the options\b',
    ]

    for line in lines:
        low = line.lower()
        if any(re.search(p, low, re.IGNORECASE) for p in drop_patterns):
            continue
        kept.append(line)

    cleaned = "\n".join(kept)

    cleaned = re.sub(r'option\s*[A-D]\s*[:：]?\s*[+-]?\d+\s*(?:to|-)\s*[+-]?\d+', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b[A-D]\s*:\s*[+-]?\d+\s*(?:to|-)\s*[+-]?\d+', ' ', cleaned, flags=re.IGNORECASE)

    return cleaned


def extract_counting_estimates_by_figure_robust(text: str) -> Dict[int, List[float]]:
    """
    更鲁棒地从 counting 思维链中抽取每个 figure 的人数估计。
    返回:
        {
            1: [350, 400, 375],
            2: [250],
            3: [200, 220]
        }
    """
    estimates = defaultdict(list)
    if not text:
        return {}

    text = text.lower()
    text = _remove_option_like_text(text)
    text = text.replace('–', '-').replace('—', '-').replace('−', '-')
    text = re.sub(r'\s+', ' ', text)

    fig_alias = _figure_alias_patterns()

    estimate_cues = r'(?:about|around|approximately|approx|roughly|maybe|perhaps|somewhere around|close to|at least|nearly)?'
    people_cues = r'(?:people|persons|crowd)?'

    num = r'(\d+(?:\.\d+)?)'
    num_range = r'(\d+(?:\.\d+)?)\s*(?:-|to|~)\s*(\d+(?:\.\d+)?)'

    for fig_id, alias_pat in fig_alias.items():
        patterns = [
            rf'(?:{alias_pat})\s*(?:[:=]|has|have|contains|shows|with|is|are|maybe|has about|has around)?\s*{estimate_cues}\s*{num_range}\s*{people_cues}',
            rf'(?:{alias_pat})\s*(?:[:=]|has|have|contains|shows|with|is|are|maybe|has about|has around)?\s*{estimate_cues}\s*{num}\s*{people_cues}',

            rf'in\s+(?:the\s+)?(?:{alias_pat}).{{0,40}}?{estimate_cues}\s*{num_range}\s*{people_cues}',
            rf'in\s+(?:the\s+)?(?:{alias_pat}).{{0,40}}?{estimate_cues}\s*{num}\s*{people_cues}',

            rf'(?:{alias_pat}).{{0,40}}?\b(?:has|have|contains|shows|is|are|looks like|might have|may have|could have)\b.{{0,20}}?{estimate_cues}\s*{num_range}\s*{people_cues}',
            rf'(?:{alias_pat}).{{0,40}}?\b(?:has|have|contains|shows|is|are|looks like|might have|may have|could have)\b.{{0,20}}?{estimate_cues}\s*{num}\s*{people_cues}',

            rf'(?:if|suppose|assume|let\'?s say)\s+(?:{alias_pat})\s*(?:is|has)?\s*{estimate_cues}\s*{num_range}\s*{people_cues}',
            rf'(?:if|suppose|assume|let\'?s say)\s+(?:{alias_pat})\s*(?:is|has)?\s*{estimate_cues}\s*{num}\s*{people_cues}',
        ]


        for pattern in patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                groups = m.groups()
                nums = [g for g in groups if g is not None and re.fullmatch(r'\d+(?:\.\d+)?', g)]
                if not nums:
                    continue

                val = None
                if len(nums) >= 2:
                    val = _normalize_count_value(nums[-2], nums[-1])
                elif len(nums) == 1:
                    val = _normalize_count_value(nums[-1])

                if val is None:
                    continue
                if val < 0:
                    continue

                estimates[fig_id].append(float(val))

    # 局部窗口补充抽取
    for fig_id, alias_pat in fig_alias.items():
        for m in re.finditer(alias_pat, text, re.IGNORECASE):
            start = m.end()
            window = text[start:start + 80]

            if re.search(r'\boption\b|\b[A-D]\s*:\s*[+-]?\d+', window, re.IGNORECASE):
                continue

            range_match = re.search(
                rf'(?:about|around|approximately|approx|roughly|maybe)?\s*{num_range}\s*(?:people|persons)?',
                window,
                re.IGNORECASE
            )
            if range_match:
                val = _normalize_count_value(range_match.group(1), range_match.group(2))
                if val is not None and val >= 0:
                    estimates[fig_id].append(float(val))
                    continue

            single_match = re.search(
                rf'(?:about|around|approximately|approx|roughly|maybe)?\s*{num}\s*(?:people|persons)?',
                window,
                re.IGNORECASE
            )
            if single_match:
                val = _normalize_count_value(single_match.group(1))
                if val is not None and val >= 0:
                    estimates[fig_id].append(float(val))

    return dict(estimates)


def get_final_counting_estimates(estimates: Dict[int, List[float]]) -> Dict[int, float]:
    """
    从每个 figure 的多个候选估计中得到最终估计。
    策略:
    - 若只有一个值，直接用
    - 若有多个值，优先使用后半段候选的中位数，减少前期随意猜测的影响
    """
    final_estimates = {}

    for fig_id, vals in estimates.items():
        if not vals:
            continue

        vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            continue

        if len(vals) == 1:
            final_estimates[fig_id] = vals[0]
            continue

        tail_vals = vals[len(vals) // 2:]
        final_estimates[fig_id] = float(np.median(tail_vals))

    return final_estimates


def compute_estimate_stability_factor(estimates: Dict[int, List[float]]) -> float:
    """
    根据每个 figure 的估计波动，返回 [0,1] 的稳定性系数。
    越稳定越接近 1，波动越大越接近 0。
    """
    drift_scores = []

    for fig_id in [1, 2, 3]:
        vals = estimates.get(fig_id, [])
        vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
        if len(vals) < 2:
            continue

        med = float(np.median(vals))
        if abs(med) < 1e-6:
            continue

        mad = float(np.median([abs(v - med) for v in vals]))
        rel_drift = mad / max(abs(med), 1.0)
        drift_scores.append(rel_drift)

    if not drift_scores:
        return 1.0

    avg_drift = float(np.mean(drift_scores))
    stability = float(np.exp(-avg_drift / 0.5))
    return min(1.0, max(0.0, stability))


def infer_counting_letters_from_reasoning(comp: str, question_text: str) -> str:
    """
    尝试从 counting 任务的推理文本中恢复三个子问题的答案字母。
    """
    think_text = extract_think_text(comp)
    estimates = extract_counting_estimates_by_figure_robust(think_text)
    final_estimates = get_final_counting_estimates(estimates)

    if not all(k in final_estimates for k in [1, 2, 3]):
        return ''

    try:
        v1 = final_estimates[1]
        v2 = final_estimates[2]
        v3 = final_estimates[3]

        all_options = parse_counting_question_options(question_text)
        if len(all_options) != 3:
            return ''

        d12 = v2 - v1
        d13 = v3 - v1
        d23 = v3 - v2

        l12 = get_counting_option_letter_from_value(d12, all_options[0])
        l13 = get_counting_option_letter_from_value(d13, all_options[1])
        l23 = get_counting_option_letter_from_value(d23, all_options[2])

        if all([l12, l13, l23]):
            return f"{l12}{l13}{l23}"
        return ''
    except Exception as e:
        print(f"[Warning] infer_counting_letters_from_reasoning error: {e}")
        return ''


def infer_symbolic_change(value_text: str, task_type: Optional[str] = None) -> Optional[int]:
    """
    将 increase/decrease/no change 等语义映射为 +1/0/-1。
    对 depth 与 distance 共享。
    """
    t = value_text.lower().strip()
    t = re.sub(r'\s+', ' ', t)

    if 'no change' in t or 'same' in t or 'unchanged' in t:
        return 0
    if 'decrease' in t or 'decreasing' in t or 'smaller' in t or 'less' in t or 'closer' in t:
        return -1
    if 'increase' in t or 'increasing' in t or 'larger' in t or 'more' in t or 'farther' in t or 'further' in t:
        return +1
    return None


def extract_symbolic_predictions(text: str) -> List[int]:
    """
    从文本中提取 symbolic change 序列(主要给 depth/distance 用)。
    按出现顺序抽取 decrease/increase/no change 等词。
    """
    patterns = [
        r'(no\s+change)',
        r'(decreases?)',
        r'(decreasing)',
        r'(increases?)',
        r'(increasing)',
        r'(same)',
        r'(unchanged)',
        r'(smaller)',
        r'(larger)',
        r'(less)',
        r'(more)',
        r'(closer)',
        r'(farther)',
        r'(further)',
    ]

    hits = []
    text_lower = text.lower()

    for p in patterns:
        for m in re.finditer(p, text_lower, re.IGNORECASE):
            phrase = m.group(1)
            val = infer_symbolic_change(phrase)
            if val is not None:
                hits.append((m.start(), val))

    hits.sort(key=lambda x: x[0])
    return [x[1] for x in hits]


def invert_mapping(mapping: dict, value: int) -> Optional[str]:
    """从 option->semantic_value 映射反推 semantic_value->option。"""
    for k, v in mapping.items():
        if v == value:
            return k
    return None


def infer_depth_letters_from_reasoning(comp: str, question_text: str) -> str:
    """尝试从 depth 任务推理文本中恢复答案字母。"""
    think_text = extract_think_text(comp)
    preds = extract_symbolic_predictions(think_text)

    if len(preds) < 3:
        return ''

    mappings = parse_depth_question_options(question_text)
    if len(mappings) != 3:
        return ''

    letters = []
    for i in range(3):
        letter = invert_mapping(mappings[i], preds[i])
        if not letter:
            return ''
        letters.append(letter)

    return ''.join(letters)


def infer_distance_letters_from_reasoning(comp: str, question_text: str) -> str:
    """尝试从 distance 任务推理文本中恢复答案字母。"""
    think_text = extract_think_text(comp)
    preds = extract_symbolic_predictions(think_text)

    if len(preds) < 3:
        return ''

    mappings = parse_distance_question_options(question_text)
    if len(mappings) != 3:
        return ''

    letters = []
    for i in range(3):
        letter = invert_mapping(mappings[i], preds[i])
        if not letter:
            return ''
        letters.append(letter)

    return ''.join(letters)


def compute_answer_mapping_reward(comp: str,
                                  question_text: Optional[str],
                                  gt_letters: str,
                                  task_type: str,
                                  map_reward_weight: float = 0.2) -> float:
    """
    Answer Mapping 奖励:
    - 从 reasoning 中恢复出隐含答案 letters
    - 与 gt_letters 做比对
    - 若恢复失败则返回 0
    """
    if not question_text or not gt_letters:
        return 0.0

    inferred_letters = ''

    try:
        if task_type == 'counting':
            inferred_letters = infer_counting_letters_from_reasoning(comp, question_text)
        elif task_type == 'depth':
            inferred_letters = infer_depth_letters_from_reasoning(comp, question_text)
        elif task_type == 'distance':
            inferred_letters = infer_distance_letters_from_reasoning(comp, question_text)
        elif task_type == 'relation':
            inferred_letters = ''
        else:
            inferred_letters = ''

        if not inferred_letters:
            return 0.0

        map_score, _ = answer_judge(inferred_letters, gt_letters)
        return float(map_score * map_reward_weight)

    except Exception as e:
        print(f"[Warning] compute_answer_mapping_reward error: {e}")
        return 0.0


# ==================== 属性估计奖励：Counting 专用 ====================
def get_gt_counts_from_image_label(image_label: dict) -> Optional[Tuple[float, float, float]]:
    """
    从 image_label 中提取三张图的 GT count。
    期望格式:
    {
        "image_1": {"count": 84},
        "image_2": {"count": 200},
        "image_3": {"count": 120}
    }
    """
    try:
        c1 = float(image_label["image_1"]["count"])
        c2 = float(image_label["image_2"]["count"])
        c3 = float(image_label["image_3"]["count"])
        return c1, c2, c3
    except Exception:
        return None


def compute_counting_attribute_reward(comp: str,
                                      image_label: Optional[dict],
                                      attr_reward_weight: float = 0.2,
                                      tau: float = 0.35,
                                      min_required_figures: int = 3,
                                      use_stability: bool = True) -> float:
    """
    counting 任务属性估计奖励:
    - 从 reasoning 中抽取 figure 1/2/3 的人数估计
    - 与 image_label 中 GT count 做比较
    - 相对误差越小, 奖励越高
    - 若缺少足够的 figure estimate, 返回 0

    奖励形式:
        reward = attr_reward_weight * accuracy_score * stability_factor
    """
    if not image_label:
        return 0.0

    gt_counts = get_gt_counts_from_image_label(image_label)
    if gt_counts is None:
        return 0.0

    think_text = extract_think_text(comp)
    estimates = extract_counting_estimates_by_figure_robust(think_text)
    final_estimates = get_final_counting_estimates(estimates)

    available = [k for k in [1, 2, 3] if k in final_estimates]
    if len(available) < min_required_figures:
        return 0.0

    try:
        pred_counts = {
            1: float(final_estimates[1]),
            2: float(final_estimates[2]),
            3: float(final_estimates[3]),
        }

        gt_map = {
            1: float(gt_counts[0]),
            2: float(gt_counts[1]),
            3: float(gt_counts[2]),
        }

        rel_errors = []
        for fig_id in [1, 2, 3]:
            pred = pred_counts[fig_id]
            gt = gt_map[fig_id]
            rel_err = abs(pred - gt) / max(abs(gt), 1.0)
            rel_errors.append(rel_err)

        avg_rel_err = float(np.mean(rel_errors))
        accuracy_score = float(np.exp(-avg_rel_err / max(tau, 1e-6)))
        accuracy_score = min(1.0, max(0.0, accuracy_score))

        stability_factor = 1.0
        if use_stability:
            stability_factor = compute_estimate_stability_factor(estimates)

        return float(attr_reward_weight * accuracy_score * stability_factor)

    except Exception as e:
        print(f"[Warning] compute_counting_attribute_reward error: {e}")
        return 0.0


# ==================== 统一的 ORM ====================
class UnifiedTripletORM(ORM):
    """
    统一的多任务三元组奖励函数(支持 counting, depth, distance, relation)

    根据 task_type 字段动态选择相应的逻辑检查函数:
    - counting: 计数任务,使用范围传递性检查
    - depth: 深度任务,使用深度变化传递性检查
    - distance: 距离任务,使用距离变化传递性检查
    - relation: 空间关系任务,不使用传递性检查

    包含五个奖励组件:
    1. 任务准确性 (r_task): 答案正确性评分
    2. 长度奖励 (r_len): 鼓励简洁的推理过程
    3. 逻辑一致性 (r_logic): 传递性约束检查(relation任务除外)
    4. Answer Mapping (r_map): 从 reasoning 中恢复隐含答案并与 GT 对齐
    5. 属性估计奖励 (r_attr): counting 专用, 检查 CoT 对每张图人数估计与 image_label 的差异
    """

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        # ========== 调试输出(首次运行时) ==========
        if not hasattr(self, '_debug_printed'):
            print("\n" + "=" * 60)
            print("DEBUG: kwargs 中的所有字段:")
            print("=" * 60)
            for key, value in kwargs.items():
                if isinstance(value, list):
                    print(f"  {key}: List[{len(value)}]")
                    if len(value) > 0:
                        sample = str(value[0])[:200]
                        print(f"    示例: {sample}...")
                else:
                    print(f"  {key}: {type(value).__name__}")
            print("=" * 60 + "\n")
            self._debug_printed = True

        # ========== 超参数配置 ==========
        group_size = 4
        alpha = 1.0
        lam = 0.4
        beta = 0.3
        gamma_map = 0.1
        eta_attr = 0.2
        sigma_floor = 1e-6
        quantile_threshold = 0.7
        correct_thr = 0.5

        # ========== 输入验证(兼容多种字段名) ==========
        solutions = kwargs.get("solution") or kwargs.get("solutions")
        question_texts = (
            kwargs.get("question_text") or
            kwargs.get("question_texts") or
            kwargs.get("query")
        )
        task_types = kwargs.get("task_type") or kwargs.get("task_types")
        image_labels = kwargs.get("image_label") or kwargs.get("image_labels")

        if not solutions:
            raise ValueError("Missing 'solution' field in kwargs")
        if len(solutions) != len(completions):
            raise ValueError(f"Length mismatch: solutions={len(solutions)}, completions={len(completions)}")
        if not task_types:
            raise ValueError("Missing 'task_type' field in kwargs")

        # 统一转为 List 格式
        if isinstance(solutions, str):
            solutions = [solutions] * len(completions)
        if isinstance(question_texts, str):
            question_texts = [question_texts] * len(completions)
        if isinstance(task_types, str):
            task_types = [task_types] * len(completions)

        if image_labels is None:
            image_labels = [None] * len(completions)
        elif isinstance(image_labels, dict):
            image_labels = [image_labels] * len(completions)

        answer_pattern = r'<answer>\s*([A-Za-z]+)\s*</answer>'

        # ========== 第一阶段:解析与评分 ==========
        results = []
        for idx, (comp, sol) in enumerate(zip(completions, solutions)):
            try:
                think_end_match = re.search(r'</think>\s*', comp, re.IGNORECASE | re.DOTALL)
                text_to_search = comp[think_end_match.end():] if think_end_match else comp

                match = re.search(answer_pattern, text_to_search, re.IGNORECASE)
                gt_match = re.search(answer_pattern, sol, re.IGNORECASE)

                if not match or not gt_match:
                    results.append({
                        'idx': idx,
                        'group_index': idx // group_size,
                        'r_task': 0.0,
                        'r_map': 0.0,
                        'r_attr': 0.0,
                        'gt_letters': '',
                        'L': extract_process_length(comp),
                        'r_logic': 0.0,
                        'valid': False
                    })
                    continue

                letters = match.group(1).upper()
                gt_letters = gt_match.group(1).upper()

                # 任务准确性评分
                r_task, _ = answer_judge(letters, gt_letters)

                # 提取推理过程长度
                L = extract_process_length(comp)

                # 根据 task_type 选择逻辑一致性检查函数
                r_logic = 0.0
                task_type = task_types[idx].lower() if idx < len(task_types) else ''
                q_text = question_texts[idx] if (question_texts and idx < len(question_texts)) else None
                image_label = image_labels[idx] if idx < len(image_labels) else None

                if q_text:
                    logic_score = 0

                    if task_type == 'counting':
                        logic_score = check_counting_transitivity(letters, q_text)
                    elif task_type == 'depth':
                        logic_score = check_depth_transitivity(letters, q_text)
                    elif task_type == 'distance':
                        logic_score = check_distance_transitivity(letters, q_text)
                    elif task_type == 'relation':
                        logic_score = check_spatial_consistency(letters, q_text)
                    else:
                        print(f"[Warning] Unknown task_type: {task_type} at index {idx}")

                    r_logic = logic_score * beta

                # Answer Mapping 奖励
                r_map = compute_answer_mapping_reward(
                    comp=comp,
                    question_text=q_text,
                    gt_letters=gt_letters,
                    task_type=task_type,
                    map_reward_weight=gamma_map
                )
                r_map += compute_answer_mapping_reward(
                    comp=comp,
                    question_text=q_text,
                    gt_letters=letters,
                    task_type=task_type,
                    map_reward_weight=gamma_map
                )

                # 属性估计奖励：只对 counting 生效
                r_attr = 0.0
                if task_type == 'counting':
                    r_attr = compute_counting_attribute_reward(
                        comp=comp,
                        image_label=image_label,
                        attr_reward_weight=eta_attr,
                        tau=0.35,
                        min_required_figures=3,
                        use_stability=True
                    )

                results.append({
                    'idx': idx,
                    'group_index': idx // group_size,
                    'r_task': float(r_task),
                    'r_map': float(r_map),
                    'r_attr': float(r_attr),
                    'gt_letters': gt_letters,
                    'r_logic': float(r_logic),
                    'L': L,
                    'valid': True
                })

            except Exception as e:
                print(f"[Warning] Error processing completion {idx}: {e}")
                results.append({
                    'idx': idx,
                    'group_index': idx // group_size,
                    'r_task': 0.0,
                    'r_map': 0.0,
                    'r_attr': 0.0,
                    'r_logic': 0.0,
                    'gt_letters': '',
                    'L': extract_process_length(comp),
                    'valid': False
                })

        # ========== 第二阶段:分组处理 ==========
        groups = defaultdict(list)
        for r in results:
            groups[r['group_index']].append(r)

        rewards = [0.0] * len(completions)

        for g_idx, group in groups.items():
            # 检查组内ground truth一致性
            if not ensure_group_solution_consistency_by_gt(group, rewards, g_idx):
                continue

            # 检查是否有有效样本
            if not any(r['valid'] for r in group):
                continue

            # 筛选高性能样本集合 H
            r_tasks = np.array([r['r_task'] for r in group], dtype=float)
            thr0 = float(np.quantile(r_tasks, quantile_threshold))
            thr = max(correct_thr, thr0)

            H = [r for r in group if r['r_task'] > thr and r['valid']]
            if len(H) == 0:
                K = max(1, len(group) // 2)
                H = sorted(
                    [r for r in group if r['valid']],
                    key=lambda x: x['r_task'],
                    reverse=True
                )[:K]

            # 计算长度分布参数
            lengths_H = np.array([r['L'] for r in H], dtype=float)
            mu_L = float(np.mean(lengths_H))
            sigma_L = float(np.std(lengths_H))
            sigma2 = max(sigma_L ** 2, sigma_floor)

            # ========== 第三阶段:综合奖励计算 ==========
            for r in group:
                L_i = float(r['L'])

                # 长度奖励(高斯衰减)
                r_len_i = alpha * np.exp(- (L_i - mu_L) ** 2 / (2.0 * sigma2))

                # 综合奖励 = 任务准确性 + 长度奖励 + 逻辑一致性 + Answer Mapping + 属性估计奖励
                total_reward = float(
                    r['r_task'] +
                    lam * r_len_i +
                    r.get('r_logic', 0.0) +
                    r.get('r_map', 0.0) +
                    r.get('r_attr', 0.0)
                )

                rewards[r['idx']] = total_reward

        return rewards


# 注册统一的 ORM
orms['unified_triplet_orm'] = UnifiedTripletORM