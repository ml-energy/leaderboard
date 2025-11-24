import { IndexData, TaskData, ModelDetail } from '../types';

const DATA_BASE_PATH = `${import.meta.env.BASE_URL}data`;

export async function loadIndexData(): Promise<IndexData> {
  const response = await fetch(`${DATA_BASE_PATH}/index.json`);
  if (!response.ok) {
    throw new Error(`Failed to load index data: ${response.statusText}`);
  }
  return response.json();
}

export async function loadTaskData(task: string): Promise<TaskData> {
  const response = await fetch(`${DATA_BASE_PATH}/tasks/${task}.json`);
  if (!response.ok) {
    throw new Error(`Failed to load task data for ${task}: ${response.statusText}`);
  }
  return response.json();
}

export async function loadModelDetail(detailFile: string): Promise<ModelDetail> {
  const response = await fetch(`${DATA_BASE_PATH}/${detailFile}`);
  if (!response.ok) {
    throw new Error(`Failed to load model detail: ${response.statusText}`);
  }
  return response.json();
}
