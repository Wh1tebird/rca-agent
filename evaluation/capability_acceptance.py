"""Build the final acceptance report from frozen fixed/AI diagnostic artifacts."""
import json
from pathlib import Path

from evaluation.evidence_coverage import evidence_coverage


def fraction(n, d):
    return f"{n}/{d} ({n / d:.1%})" if d else "0/0 (N/A)"


def main():
    root = Path("analysis")
    groups = {}
    rows = ["# Capability 计划最终验收", "",
            ("两组使用相同 12 Case、deepseek-chat、关闭 thinking、20 次查询预算、30 轮上限。"
            "fixed 保持原五指标采样；ai 在相同最多十次采样查询预算内选择指标或实体探索，"
            "最多两次选择调用。两组均在答案冻结后读取 GT。"), "",
            "| 指标 | fixed | ai |", "|---|---:|---:|"]
    summaries = {}
    for mode in ("fixed", "ai"):
        summary = json.loads((root / f"capability_final_{mode}_summary.json").read_text())
        assert summary["summary"]["completed_cases"] == 12
        assert summary["summary"]["execution_failures"] == 0
        summaries[mode] = summary["summary"]
        groups[mode] = {}
        for case in summary["cases"]:
            result = json.loads((root / f"capability_final_{mode}" / f"{case['case_id']}.json")
                                .read_text())
            groups[mode][case["case_id"]] = evidence_coverage(result)
    for metric in ("raw_gt_mechanism_proposed", "compiled", "correct_verified",
                   "wrong_verified", "replay_valid", "alternative_resolution_failure_cases"):
        rows.append(f"| {metric} | {summaries['fixed'][metric]} | {summaries['ai'][metric]} |")
    for label, numerator, denominator in (
        ("查询成功率", "successful_queries", "queries"),
        ("返回可信观测的查询比例", "queries_with_trusted_observations", "queries"),
        ("有效证据覆盖率", "checks_with_eligible_evidence", "required_checks"),
        ("竞争解释解决率", "resolved_alternatives", "declared_alternatives"),
    ):
        values = [fraction(sum(c[numerator] for c in groups[mode].values()),
                           sum(c[denominator] for c in groups[mode].values()))
                  for mode in ("fixed", "ai")]
        rows.append(f"| {label} | {values[0]} | {values[1]} |")
    rows += ["", ("有效证据覆盖分母为编译候选要求的检查数，分子为候选明确绑定、通过实体、"
             "查询指纹、可信度、完整性和样本条件检查的证据数；使用 Core 的 assess_dynamic "
             "重算。证据可用不等于预测成立。竞争解释按候选声明计数，同名解释属于不同候选时"
             "分别计数。旧历史运行未保存这些输入，其覆盖率不补算为零。"), "",
             "## 逐 Case 覆盖", "",
             "| 模式 | Case | 查询成功 | 有效证据 | 竞争解释解决 |",
             "|---|---|---:|---:|---:|"]
    for mode, cases in groups.items():
        for case, c in cases.items():
            rows.append(f"| {mode} | {case} | {fraction(c['successful_queries'], c['queries'])} | "
                        f"{fraction(c['checks_with_eligible_evidence'], c['required_checks'])} | "
                        f"{fraction(c['resolved_alternatives'], c['declared_alternatives'])} |")
    rows += ["", ("单轮模型结果存在波动，不把两组差异直接解释为因果增益。语义复核、"
             "逐案失败归因和计划验收清单见 capability_semantic_review.md 与 "
             "capability_plan_acceptance.md。")]
    (root / "capability_final_coverage.json").write_text(json.dumps(groups, indent=2) + "\n")
    (root / "capability_final_acceptance.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
