import { IndexData, TaskData, ModelDetail } from '../types';

const DATA_BASE_PATH = `${import.meta.env.BASE_URL}data`;

const cache = new Map<string, unknown>();

async function fetchJson<T>(url: string): Promise<T> {
  const cached = cache.get(url);
  if (cached !== undefined) return cached as T;

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.statusText}`);
  }
  const data = await response.json();
  cache.set(url, data);
  return data as T;
}

export async function loadIndexData(): Promise<IndexData> {
  return fetchJson<IndexData>(`${DATA_BASE_PATH}/index.json`);
}

export async function loadTaskData(task: string): Promise<TaskData> {
  return fetchJson<TaskData>(`${DATA_BASE_PATH}/tasks/${task}.json`);
}

export async function loadModelDetail(detailFile: string): Promise<ModelDetail> {
  return fetchJson<ModelDetail>(`${DATA_BASE_PATH}/${detailFile}`);
}

export function prefetchAllTasks(tasks: string[]): void {
  const schedule = typeof requestIdleCallback === 'function'
    ? requestIdleCallback
    : (cb: () => void) => setTimeout(cb, 100);

  schedule(() => {
    for (const task of tasks) {
      loadTaskData(task).catch(() => {});
    }
  });
}
