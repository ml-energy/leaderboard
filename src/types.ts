export interface ModelInfo {
  nickname: string;
  total_params_billions: number;
  activated_params_billions: number;
  is_moe: boolean;
  weight_precision: string;
}

export interface IndexData {
  last_updated: string;
  tasks: string[];
  models: Record<string, ModelInfo>;
}

export interface Configuration {
  model_id: string;
  nickname: string;
  gpu_model: string;
  num_gpus: number;
  total_params_billions: number;
  activated_params_billions: number;
  max_num_seqs: number | null;
  max_num_batched_tokens: number | null;

  energy_per_token_joules: number;
  energy_per_request_joules: number;
  median_itl_ms: number;
  output_throughput_tokens_per_sec: number;

  p95_itl_ms: number;
  avg_batch_size: number;
}

export interface TaskData {
  task: string;
  task_display_name: string;
  configurations: Configuration[];
}

export interface Timeline {
  timestamps: number[];
  values: number[];
}

export interface Parallelization {
  tensor_parallel: number;
  expert_parallel: number;
  data_parallel: number;
  notes: string;
}

export interface ConfigWithTimelines {
  gpu_model: string;
  num_gpus: number;
  max_num_seqs: number | null;
  max_num_batched_tokens: number | null;
  parallelization: Parallelization;

  energy_per_token_joules: number;
  energy_per_request_joules: number;
  median_itl_ms: number;
  p95_itl_ms: number;
  output_throughput_tokens_per_sec: number;
  avg_batch_size: number;

  timelines: {
    power_instant?: Timeline;
    power_average?: Timeline;
    batch_size?: Timeline;
  };
}

export interface Distribution {
  bins: number[];
  counts: number[];
}

export interface ModelDetail {
  model_id: string;
  task: string;
  total_params_billions: number;
  activated_params_billions: number;
  is_moe: boolean;
  weight_precision: string;

  output_length_distribution: Distribution;

  configurations: ConfigWithTimelines[];
}
