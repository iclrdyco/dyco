"""Binary accuracy and format rewards for GRPO training."""

import re


def extract_answer(text):
    matches = re.findall(r"<answer>(.*?)</answer>", str(text), flags=re.DOTALL)
    return (matches[-1] if matches else str(text)).strip()


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    return completion[0]["content"]


def answers_match(prediction, reference):
    prediction, reference = extract_answer(prediction), extract_answer(reference)
    if not prediction or not reference:
        return False
    if re.sub(r"\s+", "", prediction) == re.sub(r"\s+", "", reference):
        return True
    # Compare an explicit multiple-choice answer only when the reference is a letter.
    gold_choice = re.fullmatch(r"\(?([A-Z])\)?[.)]?", reference)
    if gold_choice:
        pred_choice = re.match(r"^\(?([A-Z])\)?(?:[.)](?:\s|$)|\s|$)", prediction)
        return pred_choice is not None and pred_choice.group(1) == gold_choice.group(1)
    from math_verify import parse, verify
    try:
        gold = parse(reference)
        predicted = parse(prediction)
        return bool(gold and predicted and verify(gold, predicted))
    except (ValueError, TypeError, ArithmeticError):
        return False


def accuracy_reward(completions, solution, **kwargs):
    if len(completions) != len(solution):
        raise ValueError("Completion and reference batch sizes differ.")
    return [float(answers_match(completion_text(c), s)) for c, s in zip(completions, solution)]


def format_reward(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    return [float(re.fullmatch(pattern, completion_text(c).strip(), re.DOTALL) is not None) for c in completions]
