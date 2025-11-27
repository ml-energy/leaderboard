export interface ColumnDef {
  key: string;
  label: string;
  sortable: boolean;
  format?: (value: any) => string;
  /** For computed columns: extract value from config object */
  getValue?: (config: any) => any;
}

export const DEFAULT_COLUMNS: ColumnDef[] = [
  { key: 'nickname', label: 'Model', sortable: true },
  {
    key: 'total_params_billions',
    label: 'Total Params (B)',
    sortable: true,
    format: (v: number) => v.toFixed(1)
  },
  {
    key: 'activated_params_billions',
    label: 'Active Params (B)',
    sortable: true,
    format: (v: number) => v.toFixed(1)
  },
  { key: 'gpu_model', label: 'GPU', sortable: true },
  { key: 'num_gpus', label: '# GPUs', sortable: true },
  {
    key: 'energy_per_token_joules',
    label: 'Energy/Token (J)',
    sortable: true,
    format: (v: number) => v.toFixed(4)
  },
  {
    key: 'energy_per_request_joules',
    label: 'Energy/Request (J)',
    sortable: true,
    format: (v: number) => v.toFixed(2)
  },
  {
    key: 'median_itl_ms',
    label: 'Median ITL (ms)',
    sortable: true,
    format: (v: number) => v.toFixed(1)
  },
  {
    key: 'output_throughput_tokens_per_sec',
    label: 'Throughput (tok/s)',
    sortable: true,
    format: (v: number) => v.toFixed(0)
  },
];

export const ADVANCED_COLUMNS: ColumnDef[] = [
  { key: 'max_num_seqs', label: 'Max Batch Size', sortable: true },
  {
    key: 'avg_batch_size',
    label: 'Avg Batch Size',
    sortable: true,
    format: (v: number) => v.toFixed(1)
  },
  {
    key: 'p95_itl_ms',
    label: 'P95 ITL (ms)',
    sortable: true,
    format: (v: number) => v.toFixed(1)
  },
];

// Diffusion model columns (text-to-image)
export const IMAGE_COLUMNS: ColumnDef[] = [
  { key: 'nickname', label: 'Model', sortable: true },
  {
    key: 'total_params_billions',
    label: 'Params',
    sortable: true,
    format: (v: number) => v < 1 ? `${(v * 1000).toFixed(0)}M` : `${v.toFixed(1)}B`
  },
  { key: 'gpu_model', label: 'GPU', sortable: true },
  { key: 'num_gpus', label: '# GPUs', sortable: true },
  { key: 'batch_size', label: 'Batch Size', sortable: true },
  {
    key: 'energy_per_image_joules',
    label: 'Energy/Image (J)',
    sortable: true,
    format: (v: number) => v.toFixed(1)
  },
  {
    key: 'batch_latency_s',
    label: 'Batch Latency (s)',
    sortable: true,
    format: (v: number) => v.toFixed(2)
  },
  {
    key: 'throughput_images_per_sec',
    label: 'Throughput (img/s)',
    sortable: true,
    format: (v: number) => v.toFixed(3)
  },
  {
    key: 'resolution',
    label: 'Resolution',
    sortable: false,
    getValue: (config: any) => `${config.image_height}×${config.image_width}`
  },
];

export const IMAGE_ADVANCED_COLUMNS: ColumnDef[] = [
  { key: 'inference_steps', label: 'Steps', sortable: true },
  { key: 'ulysses_degree', label: 'Ulysses', sortable: true },
  { key: 'ring_degree', label: 'Ring', sortable: true },
];

// Diffusion model columns (text-to-video)
export const VIDEO_COLUMNS: ColumnDef[] = [
  { key: 'nickname', label: 'Model', sortable: true },
  {
    key: 'total_params_billions',
    label: 'Params',
    sortable: true,
    format: (v: number) => v < 1 ? `${(v * 1000).toFixed(0)}M` : `${v.toFixed(1)}B`
  },
  { key: 'gpu_model', label: 'GPU', sortable: true },
  { key: 'num_gpus', label: '# GPUs', sortable: true },
  { key: 'batch_size', label: 'Batch Size', sortable: true },
  {
    key: 'energy_per_video_joules',
    label: 'Energy/Video (J)',
    sortable: true,
    format: (v: number) => v.toFixed(1)
  },
  {
    key: 'batch_latency_s',
    label: 'Batch Latency (s)',
    sortable: true,
    format: (v: number) => v.toFixed(2)
  },
  {
    key: 'throughput_videos_per_sec',
    label: 'Throughput (vid/s)',
    sortable: true,
    format: (v: number) => v.toFixed(4)
  },
  {
    key: 'resolution',
    label: 'Resolution',
    sortable: false,
    getValue: (config: any) => `${config.video_height}×${config.video_width}`
  },
];

export const VIDEO_ADVANCED_COLUMNS: ColumnDef[] = [
  { key: 'inference_steps', label: 'Steps', sortable: true },
  { key: 'ulysses_degree', label: 'Ulysses', sortable: true },
  { key: 'ring_degree', label: 'Ring', sortable: true },
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
  return { default: DEFAULT_COLUMNS, advanced: ADVANCED_COLUMNS };
}
