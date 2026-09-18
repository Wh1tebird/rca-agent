"""Audit aggregate-counter and denominator semantics without loading labels."""
import json
from pathlib import Path

import duckdb


def main():
    source = Path.home() / "study/AgenticOpsEval/RCA100/cases"
    with duckdb.connect() as connection:
        connection.execute("SET enable_progress_bar=false")
        rows = connection.execute("""
            WITH series AS (
                SELECT filename, entity_name, metric, time, value,
                       lag(value) OVER (
                           PARTITION BY filename,entity_name,metric ORDER BY time
                       ) AS previous
                FROM read_parquet(?, filename=true)
                WHERE metric IN (
                    'arms_thread_pool_completed_task_count',
                    'arms_thread_pool_scheduled_task_count',
                    'arms_thread_pool_core_pool_size',
                    'arms_thread_pool_current_thread_count',
                    'arms_thread_pool_max_pool_size',
                    'arms_grouped_thread_waiting_count',
                    'arms_grouped_thread_timed_waiting_count',
                    'arms_jvm_gc_total', 'arms_jvm_gc_seconds_total', 'pod_restart_count'
                )
            )
            SELECT metric, count(*), count(DISTINCT filename), min(value), max(value),
                   count(*) FILTER (WHERE value < previous),
                   count(*) FILTER (WHERE value != floor(value)),
                   count(*) FILTER (WHERE value < 0)
            FROM series GROUP BY metric ORDER BY metric
        """, [str(source / "*" / "metrics.parquet")]).fetchall()
        alignment = connection.execute("""
            WITH aligned AS (
                SELECT filename, entity_name, time,
                    max(value) FILTER (WHERE metric='error') AS errors,
                    max(value) FILTER (WHERE metric='workload') AS requests
                FROM read_parquet(?, filename=true)
                WHERE entity_set='apm.service.legacy' AND metric IN ('error','workload')
                GROUP BY filename,entity_name,time
            )
            SELECT count(*), count(*) FILTER (WHERE errors IS NULL),
                count(*) FILTER (WHERE requests IS NULL),
                count(*) FILTER (WHERE errors > requests),
                count(*) FILTER (WHERE requests = 0)
            FROM aligned
        """, [str(source / "*" / "metrics.parquet")]).fetchone()
    records = [dict(zip(("metric", "samples", "cases", "minimum", "maximum",
                        "decreases", "fractional_samples", "negative_samples"), r, strict=True))
               for r in rows]
    payload = {"counters": records, "service_error_alignment": dict(zip(
        ("timestamps", "missing_errors", "missing_requests", "errors_exceed_requests",
         "zero_requests"), alignment, strict=True))}
    Path("analysis/metric_semantics_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# 原始指标语义审计", "", "审计仅读取观测数据，不读取 GT。", "",
             "| 指标 | 样本 | Case | 下降次数 | 小数样本 | 负值样本 |",
             "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['metric']} | {r['samples']} | {r['cases']} | {r['decreases']} | "
              f"{r['fractional_samples']} | {r['negative_samples']} |" for r in records]
    lines += ["", "## 派生处理决定", "",
              ("原始字段存在不代表它是单进程累计计数。下降可能来自重启、实例聚合变化或源端"
              "窗口统计；当前数据没有过程身份/重置标记，无法可靠区分。保留原始值，明确标记"
              "任务完成速率、重启速率与线程池利用率的派生语义不可用。已提供的 GC delta 继续"
              "使用源端增量，不再对它求差分。"), "",
              "服务错误量与请求量按同实体、同时间戳检查：", "",
              "```json", json.dumps(payload["service_error_alignment"], indent=2), "```", "",
              ("缺失错误量不能解释为零错误。当前未证明两者的窗口/聚合口径一致，因此不合成"
              "服务级错误率；原生 operation/instance 错误率可查询并可做基线比较。"
              "这属于明确的派生能力不可用状态，而非将缺失输入变成零值。")]
    Path("analysis/metric_semantics_audit.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
