export interface ColumnDef {
  key: string;
  label: string;
  sortable: boolean;
  format?: (value: any) => string;
  /** For computed columns: extract value from config object */
  getValue?: (config: any) => any;
  /** Text alignment for the column */
  align?: 'left' | 'right';
  /** Tooltip text for help icon */
  tooltip?: string;
}

export const LLM_COLUMNS: ColumnDef[] = [
  { key: 'nickname', label: 'Model', sortable: true },
  {
    key: 'weight_precision',
    label: 'Precision',
    sortable: true,
    tooltip: 'The numerical precision used to store the majority of model weights (e.g., large layers like MoE experts and MLPs). Lower precision like FP8 reduces memory usage and can improve speed, while higher precision like BF16 may offer better accuracy.'
  },
  {
    key: 'total_params_billions',
    label: 'Total Params (B)',
    sortable: true,
    format: (v: number) => v.toFixed(0),
    align: 'right',
    tooltip: 'The total number of parameters in the model, measured in billions.'
  },
  {
    key: 'activated_params_billions',
    label: 'Active Params (B)',
    sortable: true,
    format: (v: number) => v.toFixed(0),
    align: 'right',
    tooltip: 'The number of parameters activated per token. For Mixture-of-Experts (MoE) models, only a subset of parameters are used for each token, making this smaller than total parameters.'
  },
  {
    key: 'gpu_model',
    label: 'GPU',
    sortable: true,
    tooltip: 'The GPU model used for inference. Different GPUs have varying performance characteristics and energy efficiency.'
  },
  {
    key: 'num_gpus',
    label: '# GPUs',
    sortable: true,
    align: 'right',
    tooltip: 'The number of GPUs used in parallel. More GPUs can handle larger models and higher throughput, but also consume more energy.'
  },
  {
    key: 'energy_per_token_joules',
    label: 'Energy/Token (J)',
    sortable: true,
    format: (v: number) => v.toFixed(4),
    align: 'right',
    tooltip: 'Energy consumed to generate one token. A token is the basic unit of text generation for large language models, roughly corresponding to a word or part of a word.'
  },
  {
    key: 'energy_per_request_joules',
    label: 'Energy/Response (J)',
    sortable: true,
    format: (v: number) => v.toFixed(2),
    align: 'right',
    tooltip: 'Energy consumed to generate a complete response. This accounts for multiple tokens and reflects both the energy efficiency and verbosity of the model. This is what matters most for end users who care about the full output.'
  },
  {
    key: 'median_itl_ms',
    label: 'Median ITL (ms)',
    sortable: true,
    format: (v: number) => v.toFixed(1),
    align: 'right',
    tooltip: 'Inter-Token Latency: the time between generating consecutive tokens. Lower values mean faster, more responsive text generation.'
  },
];

export const LLM_ADVANCED_COLUMNS: ColumnDef[] = [
  {
    key: 'avg_power_watts',
    label: 'Avg Power (W)',
    sortable: true,
    format: (v: number) => v.toFixed(0),
    align: 'right',
    tooltip: 'Average GPU power consumption during inference. Lower power at similar performance indicates better energy efficiency.'
  },
  {
    key: 'architecture',
    label: 'Architecture',
    sortable: true,
    tooltip: 'The model architecture type. "Dense" models use all parameters for every token. "MoE" (Mixture-of-Experts) models selectively activate only a subset of parameters, allowing for larger total capacity with lower compute cost.'
  },
  {
    key: 'parallelization',
    label: 'Parallelization',
    sortable: false,
    getValue: (config: any) => {
      const parts: string[] = [];
      if (config.tensor_parallel && config.tensor_parallel > 1) {
        parts.push(`TP${config.tensor_parallel}`);
      }
      if (config.expert_parallel && config.expert_parallel > 1) {
        parts.push(`EP${config.expert_parallel}`);
      }
      if (config.data_parallel && config.data_parallel > 1) {
        parts.push(`DP${config.data_parallel}`);
      }
      return parts.length > 0 ? parts.join(' ') : '-';
    },
    tooltip: 'How the model is distributed across GPUs. TP (Tensor Parallel) splits layers across GPUs. EP (Expert Parallel) distributes MoE experts across GPUs. DP (Data Parallel) replicates layers across GPUs. Some models use hybrid approaches (e.g., DP for attention + EP for MLP).'
  },
  {
    key: 'output_throughput_tokens_per_sec',
    label: 'Throughput (tok/s)',
    sortable: true,
    format: (v: number) => v.toFixed(0),
    align: 'right',
    tooltip: 'The total number of tokens generated per second across all concurrent requests. Higher throughput means the system can serve more users simultaneously.'
  },
  {
    key: 'max_num_seqs',
    label: 'Max Batch Size',
    sortable: true,
    align: 'right',
    tooltip: 'The maximum number of requests processed simultaneously. Larger batch sizes improve throughput and energy efficiency but may increase latency for individual requests.'
  },
];

// Diffusion model columns (text-to-image)
export const IMAGE_COLUMNS: ColumnDef[] = [
  { key: 'nickname', label: 'Model', sortable: true },
  {
    key: 'total_params_billions',
    label: 'Params',
    sortable: true,
    format: (v: number) => v < 1 ? `${(v * 1000).toFixed(0)}M` : `${v.toFixed(0)}B`,
    align: 'right'
  },
  { key: 'gpu_model', label: 'GPU', sortable: true },
  { key: 'num_gpus', label: '# GPUs', sortable: true, align: 'right' },
  { key: 'batch_size', label: 'Batch Size', sortable: true, align: 'right' },
  {
    key: 'energy_per_image_joules',
    label: 'Energy/Image (J)',
    sortable: true,
    format: (v: number) => v.toFixed(1),
    align: 'right'
  },
  {
    key: 'batch_latency_s',
    label: 'Batch Latency (s)',
    sortable: true,
    format: (v: number) => v.toFixed(2),
    align: 'right'
  },
  {
    key: 'throughput_images_per_sec',
    label: 'Throughput (img/s)',
    sortable: true,
    format: (v: number) => v.toFixed(3),
    align: 'right'
  },
  {
    key: 'resolution',
    label: 'Resolution',
    sortable: false,
    getValue: (config: any) => `${config.image_height}×${config.image_width}`,
    align: 'right',
    tooltip: 'The output image resolution in pixels (height × width).'
  },
  {
    key: 'inference_steps',
    label: 'Denoising Steps',
    sortable: true,
    align: 'right',
    tooltip: 'The number of denoising steps used during image generation. More steps generally improve quality but increase generation time and energy consumption.'
  },
];

export const IMAGE_ADVANCED_COLUMNS: ColumnDef[] = [
  {
    key: 'avg_power_watts',
    label: 'Avg Power (W)',
    sortable: true,
    format: (v: number) => v.toFixed(0),
    align: 'right',
    tooltip: 'Average GPU power consumption during inference. Lower power at similar performance indicates better energy efficiency.'
  },
  { key: 'ulysses_degree', label: 'Ulysses', sortable: true, align: 'right' },
  { key: 'ring_degree', label: 'Ring Attention', sortable: true, align: 'right' },
];

// Diffusion model columns (text-to-video)
export const VIDEO_COLUMNS: ColumnDef[] = [
  { key: 'nickname', label: 'Model', sortable: true },
  {
    key: 'total_params_billions',
    label: 'Params',
    sortable: true,
    format: (v: number) => v < 1 ? `${(v * 1000).toFixed(0)}M` : `${v.toFixed(0)}B`,
    align: 'right'
  },
  { key: 'gpu_model', label: 'GPU', sortable: true },
  { key: 'num_gpus', label: '# GPUs', sortable: true, align: 'right' },
  { key: 'batch_size', label: 'Batch Size', sortable: true, align: 'right' },
  {
    key: 'energy_per_video_joules',
    label: 'Energy/Video (J)',
    sortable: true,
    format: (v: number) => v.toFixed(1),
    align: 'right'
  },
  {
    key: 'batch_latency_s',
    label: 'Batch Latency (s)',
    sortable: true,
    format: (v: number) => v.toFixed(2),
    align: 'right'
  },
  {
    key: 'throughput_videos_per_sec',
    label: 'Throughput (vid/s)',
    sortable: true,
    format: (v: number) => v.toFixed(4),
    align: 'right'
  },
  {
    key: 'resolution',
    label: 'Resolution',
    sortable: false,
    getValue: (config: any) => `${config.video_height}×${config.video_width}`,
    align: 'right',
    tooltip: 'The output video resolution in pixels (height × width).'
  },
  {
    key: 'num_frames',
    label: '# Frames',
    sortable: true,
    align: 'right',
    tooltip: 'The number of frames in the generated video. More frames means longer video duration.'
  },
  {
    key: 'inference_steps',
    label: 'Denoising Steps',
    sortable: true,
    align: 'right',
    tooltip: 'The number of denoising steps used during video generation. More steps generally improve quality but increase generation time and energy consumption.'
  },
];

export const VIDEO_ADVANCED_COLUMNS: ColumnDef[] = [
  {
    key: 'avg_power_watts',
    label: 'Avg Power (W)',
    sortable: true,
    format: (v: number) => v.toFixed(0),
    align: 'right',
    tooltip: 'Average GPU power consumption during inference. Lower power at similar performance indicates better energy efficiency.'
  },
  { key: 'ulysses_degree', label: 'Ulysses', sortable: true, align: 'right' },
  { key: 'ring_degree', label: 'Ring', sortable: true, align: 'right' },
];

/**
 * Get columns for a specific task.
 */
export function getColumnsForTask(taskId: string): { default: ColumnDef[], advanced: ColumnDef[] } {
  if (taskId === 'text-to-image') {
    return { default: IMAGE_COLUMNS, advanced: IMAGE_ADVANCED_COLUMNS };
  }
  if (taskId === 'text-to-video') {
    return { default: VIDEO_COLUMNS, advanced: VIDEO_ADVANCED_COLUMNS };
  }
  return { default: LLM_COLUMNS, advanced: LLM_ADVANCED_COLUMNS };
}
