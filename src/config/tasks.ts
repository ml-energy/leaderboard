/**
 * Task-specific configuration for the leaderboard.
 * Edit this file to customize task display names, slider defaults, and about content.
 */

import type { Architecture } from '../types';

export interface TaskConfig {
  /** Display name shown in the page header */
  displayName: string;
  /** Tab label shown in the task tabs */
  tabLabel: string;
  /** Architecture type for the task */
  architecture: Architecture;
  /** Default ITL deadline in ms (for LLM/MLLM) */
  defaultItlDeadlineMs?: number;
  /** Default batch latency deadline in seconds (for diffusion) */
  defaultLatencyDeadlineS?: number;
  /** Default energy budget in J (null = use max from data) */
  defaultEnergyBudgetJ: number | null;
  /** Short description for the task */
  description?: string;
  /** About page content (markdown or HTML) */
  aboutContent?: string;
}

export const TASK_CONFIGS: Record<string, TaskConfig> = {
  // LLM tasks
  "gpqa": {
    displayName: "GPQA Diamond",
    tabLabel: "Problem Solving with Reasoning",
    architecture: "llm",
    defaultItlDeadlineMs: 500,
    defaultEnergyBudgetJ: null,
    description: "Graduate-level science questions requiring deep reasoning",
    aboutContent: `
      <section class="space-y-4">
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">Overview</h3>
          <p class="text-gray-700 dark:text-gray-300">
            GPQA (Graduate-level Google-Proof Q&A) Diamond is a benchmark consisting of challenging multiple-choice questions
            in biology, physics, and chemistry, written by domain experts. The "Diamond" subset contains the most difficult
            questions that even experts find challenging.
          </p>
        </div>
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">Benchmark Details</h3>
          <ul class="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
            <li><strong>Dataset:</strong> GPQA Diamond subset (448 questions)</li>
            <li><strong>Task type:</strong> Multiple-choice question answering with reasoning</li>
            <li><strong>Evaluation:</strong> Accuracy on selecting correct answer</li>
          </ul>
        </div>
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">Why This Task?</h3>
          <p class="text-gray-700 dark:text-gray-300">
            GPQA tests deep reasoning capabilities and domain knowledge. Models that perform well on GPQA typically
            demonstrate strong reasoning chains and accurate recall of scientific concepts. This makes it an excellent
            benchmark for evaluating energy efficiency in reasoning-intensive workloads.
          </p>
        </div>
      </section>
    `,
  },
  "lm-arena-chat": {
    displayName: "LLM Chat (LM Arena)",
    tabLabel: "Text Conversation",
    architecture: "llm",
    defaultItlDeadlineMs: 500,
    defaultEnergyBudgetJ: null,
    description: "General conversational benchmark from LM Arena",
  },
  "sourcegraph-fim": {
    displayName: "Fill-in-the-Middle (Sourcegraph)",
    tabLabel: "Code Completion",
    architecture: "llm",
    defaultItlDeadlineMs: 500,
    defaultEnergyBudgetJ: null,
    description: "Code completion benchmark from Sourcegraph",
  },
  // MLLM tasks
  "image-chat": {
    displayName: "Image Chat",
    tabLabel: "Image Chat",
    architecture: "mllm",
    defaultItlDeadlineMs: 500,
    defaultEnergyBudgetJ: null,
    description: "Multimodal image understanding and chat",
  },
  "video-chat": {
    displayName: "Video Chat",
    tabLabel: "Video Chat",
    architecture: "mllm",
    defaultItlDeadlineMs: 500,
    defaultEnergyBudgetJ: null,
    description: "Multimodal video understanding and chat",
  },
  "audio-chat": {
    displayName: "Audio Chat",
    tabLabel: "Audio Chat",
    architecture: "mllm",
    defaultItlDeadlineMs: 500,
    defaultEnergyBudgetJ: null,
    description: "Multimodal audio understanding and chat",
  },
  // Diffusion tasks
  "text-to-image": {
    displayName: "Text to Image",
    tabLabel: "Text to Image",
    architecture: "diffusion",
    defaultLatencyDeadlineS: 10,
    defaultEnergyBudgetJ: null,
    description: "Generate images from text prompts",
  },
  "text-to-video": {
    displayName: "Text to Video",
    tabLabel: "Text to Video",
    architecture: "diffusion",
    defaultLatencyDeadlineS: 120,
    defaultEnergyBudgetJ: null,
    description: "Generate videos from text prompts",
  },
};

/**
 * Get config for a task, with fallback defaults for unknown tasks.
 */
export function getTaskConfig(taskId: string): TaskConfig {
  return TASK_CONFIGS[taskId] ?? {
    displayName: taskId,
    tabLabel: taskId,
    architecture: "llm",
    defaultItlDeadlineMs: 500,
    defaultEnergyBudgetJ: null,
  };
}

/**
 * Check if a task is a diffusion task.
 */
export function isDiffusionTask(taskId: string): boolean {
  return getTaskConfig(taskId).architecture === "diffusion";
}

/**
 * Get the architecture for a task.
 */
export function getTaskArchitecture(taskId: string): Architecture {
  return getTaskConfig(taskId).architecture;
}

/**
 * Sort tasks according to TASK_CONFIGS key order.
 * Tasks not in TASK_CONFIGS appear at the end in their original order.
 */
export function sortTasks(tasks: string[]): string[] {
  const configOrder = Object.keys(TASK_CONFIGS);
  return [...tasks].sort((a, b) => {
    const indexA = configOrder.indexOf(a);
    const indexB = configOrder.indexOf(b);
    if (indexA !== -1 && indexB !== -1) return indexA - indexB;
    if (indexA !== -1) return -1;
    if (indexB !== -1) return 1;
    return 0;
  });
}
