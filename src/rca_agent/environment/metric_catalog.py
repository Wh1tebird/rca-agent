"""Host-owned mappings and public descriptions; raw source names stay behind the adapter."""

EXTRA_SERVICE_METRICS = {
    "jvm_waiting_threads": ("apm.metric.thread", "arms_grouped_thread_waiting_count", None),
    "jvm_timed_waiting_threads": (
        "apm.metric.thread", "arms_grouped_thread_timed_waiting_count", None,
    ),
    "thread_pool_core_size": ("apm.metric.jvm", "arms_thread_pool_core_pool_size", None),
    "thread_pool_current_size": (
        "apm.metric.jvm", "arms_thread_pool_current_thread_count", None,
    ),
    "thread_pool_max_size": ("apm.metric.jvm", "arms_thread_pool_max_pool_size", None),
    "thread_pool_completed_tasks": (
        "apm.metric.jvm", "arms_thread_pool_completed_task_count", None,
    ),
    "thread_pool_scheduled_tasks": (
        "apm.metric.jvm", "arms_thread_pool_scheduled_task_count", None,
    ),
}

EXTRA_DEPLOYMENT_METRICS = {
    "cpu_usage_vs_limits": ("k8s.deployment", "deployment_cpu_usage_vs_limits", None),
    "cpu_usage_vs_requests": ("k8s.deployment", "deployment_cpu_usage_vs_requests", None),
    "availability_rate": ("k8s.deployment", "deployment_availability_rate", "percent"),
    "available_replicas": ("k8s.deployment", "deployment_available_replicas", "count"),
    "desired_replicas": ("k8s.deployment", "deployment_desired_replicas", "count"),
}

REMAINING_METRICS = {
    "apm.service": {
        "exception_request_errors": (
            "apm.metric.exception", "arms_exception_requests_error_count_raw", None,
        ),
        "jvm_gc_count_total": ("apm.metric.jvm", "arms_jvm_gc_total", None),
        "jvm_gc_seconds_total": ("apm.metric.jvm", "arms_jvm_gc_seconds_total", "s"),
        "jvm_heap_committed_bytes": (
            "apm.metric.jvm", "arms_jvm_mem_committed_bytes", "bytes",
        ),
        "jvm_heap_init_bytes": ("apm.metric.jvm", "arms_jvm_mem_init_bytes", "bytes"),
        "jvm_grouped_thread_count": ("apm.metric.thread", "arms_grouped_thread_count", None),
    },
    "k8s.cluster": {
        signal: ("k8s.cluster", signal, unit)
        for signal, unit in {
            "apiserver_concurrent_connections": "count",
            "apiserver_non_2xx_read_request_qps": "requests/s",
            "apiserver_non_2xx_write_request_qps": "requests/s",
            "apiserver_requests_per_second": "requests/s",
            "cluster_failed_node_count": "count",
            "cluster_node_count": "count",
            "cluster_number_of_not_ready_pods": "count",
            "cluster_number_of_pod": "count",
        }.items()
    },
    "k8s.namespace": {
        signal: ("k8s.namespace", signal, "count")
        for signal in ("namespace_number_of_not_ready_pods", "namespace_number_of_pods")
    },
    "k8s.node": {"running_pods": ("k8s.node", "node_pod_running_count", "count")},
}

# These are desired derivations, not missing raw telemetry. Ambiguous aggregate counters
# must not be differentiated as though they were a single monotonic process counter.
UNSUPPORTED_DERIVATIONS = {
    "thread_pool_completed_tasks_per_second": "Aggregate counter/reset semantics unverified.",
    "thread_pool_utilization": "Active/max aggregate scales are incompatible.",
    "pod_restart_rate": "Source series lacks process identity and reset semantics.",
    "service_error_rate": "Service error/workload alignment is not documented; use native "
    "operation or instance error_rate where available.",
}

MISSING_SIGNALS = {
    "db_network_rtt_ms": "No transport RTT telemetry; connection acquisition is not RTT.",
    "db_server_execution_ms": "No server execution telemetry; JDBC spans include client waits.",
    "connection_pool_active": "No database connection-pool active connection telemetry.",
    "connection_pool_max": "No database connection-pool capacity telemetry.",
    "connection_pool_wait_ms": "No isolated pool queue-wait telemetry.",
    "full_gc_count": "Generic GC activity does not identify Full GC.",
    "full_gc_pause_duration": "No Full-GC-specific pause telemetry.",
    "old_gen_usage": "No generation-specific heap telemetry.",
    "allocation_rate": "No allocation-rate telemetry.",
    "heap_after_gc": "No post-GC heap telemetry.",
    "lock_wait_ms": "Aggregate waiting threads do not identify lock contention.",
    "cpu_throttling": "CPU relative to limits does not measure throttling.",
    "disk_io_wait": "Disk capacity usage does not measure IO wait.",
    "disk_latency_ms": "No disk IO latency telemetry.",
    "thread_pool_queue_wait_ms": "Queue depth does not measure queue wait duration.",
}


def metric_description(signal: str) -> str:
    if signal in EXTRA_SERVICE_METRICS:
        return (
            f"{signal}: source aggregate with unverified scale/counter semantics; "
            "context only. Do not derive pool utilization, lock causality or task rates."
        )
    if signal in {"cpu_usage_vs_limits", "cpu_usage_vs_requests"}:
        return f"{signal}: source CPU relative-to-allocation statistic; not throttling evidence."
    if signal == "pod_restart_count":
        return "Source restart count; counter/reset semantics unverified; not an OOM diagnosis."
    if signal == "request_count":
        return "Request-volume samples; compare equal windows on the same entity."
    if signal in {"error_rate", "request_latency_seconds"}:
        return f"{signal}: observed service symptom; not causal support by itself."
    return f"Bounded {signal} samples for the selected entity; missing is not zero."
