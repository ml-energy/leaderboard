export interface ColumnDef {
  key: string;
  label: string;
  sortable: boolean;
  format?: (value: any) => string;
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
