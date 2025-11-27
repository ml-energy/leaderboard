// Architecture types
export type Architecture = 'llm' | 'mllm' | 'diffusion';

export interface ModelInfo {
  nickname: string;
  total_params_billions?: number;
  activated_params_billions?: number;
  architecture?: string;
  weight_precision?: string;
}

export interface IndexData {
  last_updated: string;
  tasks: string[];
  architectures: Record<Architecture, string[]>;
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

  p90_itl_ms: number;
  p95_itl_ms: number;
  p99_itl_ms: number;
  avg_batch_size: number;
}

export interface TaskData {
  task: string;
  task_display_name: string;
  architecture?: Architecture;
  configurations: Configuration[];
}

// Diffusion-specific configuration (text-to-image)
export interface ImageConfiguration {
  model_id: string;
  nickname: string;
  gpu_model: string;
  num_gpus: number;
  total_params_billions: number;
  activated_params_billions: number;
  batch_size: number;
  energy_per_image_joules: number;
  batch_latency_s: number;
  throughput_images_per_sec: number;
  image_height: number;
  image_width: number;
  inference_steps: number;
  ulysses_degree: number;
  ring_degree: number;
}

// Diffusion-specific configuration (text-to-video)
export interface VideoConfiguration {
  model_id: string;
  nickname: string;
  gpu_model: string;
  num_gpus: number;
  total_params_billions: number;
  activated_params_billions: number;
  batch_size: number;
  energy_per_video_joules: number;
  batch_latency_s: number;
  throughput_videos_per_sec: number;
  video_height: number;
  video_width: number;
  inference_steps: number;
  ulysses_degree: number;
  ring_degree: number;
}

export interface DiffusionTaskData {
  task: string;
  task_display_name: string;
  architecture: 'diffusion';
  configurations: ImageConfiguration[] | VideoConfiguration[];
}

export interface Parallelization {
  tensor_parallel: number;
  expert_parallel: number;
  data_parallel: number;
  notes: string;
}

export interface ModelConfiguration {
  gpu_model: string;
  num_gpus: number;
  max_num_seqs: number | null;
  max_num_batched_tokens: number | null;
  parallelization: Parallelization;

  energy_per_token_joules: number;
  energy_per_request_joules: number;
  median_itl_ms: number;
  p90_itl_ms: number;
  p95_itl_ms: number;
  p99_itl_ms: number;
  output_throughput_tokens_per_sec: number;
  avg_batch_size: number;

  output_length_distribution: Distribution;
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
  architecture: string;
  weight_precision: string;

  output_length_distribution: Distribution;

  configurations: ModelConfiguration[];
}

// Diffusion model detail types
export interface DiffusionParallelization {
  ulysses_degree: number;
  ring_degree: number;
}

export interface ImageModelConfiguration {
  gpu_model: string;
  num_gpus: number;
  batch_size: number;
  parallelization: DiffusionParallelization;
  energy_per_image_joules: number;
  batch_latency_s: number;
  throughput_images_per_sec: number;
  image_height: number;
  image_width: number;
  inference_steps: number;
}

export interface VideoModelConfiguration {
  gpu_model: string;
  num_gpus: number;
  batch_size: number;
  parallelization: DiffusionParallelization;
  energy_per_video_joules: number;
  batch_latency_s: number;
  throughput_videos_per_sec: number;
  video_height: number;
  video_width: number;
  inference_steps: number;
}

export interface DiffusionModelDetail {
  model_id: string;
  task: string;
  nickname: string;
  total_params_billions: number;
  activated_params_billions: number;
  weight_precision: string;
  configurations: ImageModelConfiguration[] | VideoModelConfiguration[];
}

// Union type for any configuration
export type AnyConfiguration = Configuration | ImageConfiguration | VideoConfiguration;
export type AnyModelConfiguration = ModelConfiguration | ImageModelConfiguration | VideoModelConfiguration;
export type AnyModelDetail = ModelDetail | DiffusionModelDetail;
