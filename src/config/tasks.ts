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
    displayName: "Problem Solving with Reasoning",
    tabLabel: "Problem Solving",
    architecture: "llm",
    defaultItlDeadlineMs: 500,
    defaultEnergyBudgetJ: null,
    description: "Graduate-level science questions requiring deep reasoning",
    aboutContent: `
      <section class="space-y-6">
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">What is this?</h3>
          <p class="text-gray-700 dark:text-gray-300">
            This task represents <strong>complex problem-solving</strong>: when you ask an AI to work through a challenging math problem, debug a tricky piece of code, or explain a scientific concept step by step. The AI needs to "think" carefully and show its reasoning, often producing long, detailed explanations. We use <a href="https://arxiv.org/abs/2311.12022" target="_blank" rel="noopener noreferrer" class="text-blue-600 dark:text-blue-400 hover:underline">GPQA Diamond</a>, a set of PhD-level science questions that require deep reasoning to solve.
          </p>
        </div>
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">Going Deeper</h3>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            Problem-solving workloads have a distinctive characteristic: <strong>short inputs but very long outputs</strong>. The question itself would typically be brief, but the model generates extensive reasoning chains, sometimes tens of thousands of tokens, to work through the problem.
          </p>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            This has noticable implications for energy consumption. Long outputs mean the model's context grows throughout generation, which increases memory consumption and reduces how many responses the server can generate in parallel, or in other words, <strong>smaller batch size</strong>. When fewer requests can be processed together, the GPU's computational capacity is less efficiently utilized, making each token more expensive to generate. Combined with the sheer number of output tokens, <strong>energy per response becomes very large</strong> compared to shorter conversational tasks.
          </p>
          <p class="text-gray-700 dark:text-gray-300">
            This is why reasoning models like DeepSeek-R1 or Qwen3 with thinking mode show dramatically different energy profiles than the same base models in chat mode.
          </p>
        </div>
      </section>
    `,
  },
  "lm-arena-chat": {
    displayName: "General Text Conversation",
    tabLabel: "Text Conversation",
    architecture: "llm",
    defaultItlDeadlineMs: 500,
    defaultEnergyBudgetJ: null,
    description: "General conversational benchmark from LM Arena",
    aboutContent: `
      <section class="space-y-6">
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">What is this?</h3>
          <p class="text-gray-700 dark:text-gray-300">
            This is the most familiar AI use case: <strong>having a conversation with a chatbot</strong>. Ask a question, get help writing an email, brainstorm ideas, or get explanations on any topic. This is what services like ChatGPT, Claude, and Gemini do every day. We use real conversations from <a href="https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k" target="_blank" rel="noopener noreferrer" class="text-blue-600 dark:text-blue-400 hover:underline">Chatbot Arena</a>, where millions of users have chatted with AI models.
          </p>
        </div>
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">Going Deeper</h3>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            Text conversation represents the <strong>most balanced workload</strong> for language models. Both inputs and outputs are relatively short: a typical user message is a sentence or two, and responses are a few paragraphs at most.
          </p>
          <p class="text-gray-700 dark:text-gray-300">
            This balanced profile allows for <strong>efficient batching</strong>: the server can process many requests simultaneously, spreading the GPU's fixed overhead across more responses. This makes text conversation one of the most energy-efficient use cases per token. However, because responses are shorter, the total energy per response is also lower than reasoning-heavy tasks.
          </p>
        </div>
      </section>
    `,
  },
  "sourcegraph-fim": {
    displayName: "Code Completion with Fill-in-the-Middle",
    tabLabel: "Code Completion",
    architecture: "llm",
    defaultItlDeadlineMs: 500,
    defaultEnergyBudgetJ: null,
    description: "Code completion benchmark from Sourcegraph",
    aboutContent: `
      <section class="space-y-6">
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">What is this?</h3>
          <p class="text-gray-700 dark:text-gray-300">
            This is the <strong>inline code suggestion</strong> you see when typing in a code editor. The AI predicts what you're about to type and offers to complete it for you. Think of the "tab to accept" suggestions in AI-powered editors. The model sees the code before and after your cursor and fills in the gap.
          </p>
        </div>
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">Going Deeper</h3>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            Code completion has the <strong>opposite profile from problem-solving</strong>: inputs are long (the surrounding code context), but outputs are short (typically just a few lines of code). The model needs substantial context to understand what code would fit, but the actual completion is brief.
          </p>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            This creates an interesting energy dynamic. The <strong>prefill phase</strong> (processing the input context) dominates the computation, while the <strong>decode phase</strong> (generating the completion) is quick. Latency is critical here: code completions need to appear nearly instantly as you type, so models must be fast even with long context windows.
          </p>
          <p class="text-gray-700 dark:text-gray-300">
            The short output length keeps energy per response low, but the high frequency of completions (triggered with every keystroke in some editors) means aggregate energy consumption adds up.
          </p>
        </div>
      </section>
    `,
  },
  // MLLM tasks
  "image-chat": {
    displayName: "Image Understanding and Conversation",
    tabLabel: "Image Chat",
    architecture: "mllm",
    defaultItlDeadlineMs: 500,
    defaultEnergyBudgetJ: null,
    description: "Multimodal image understanding and chat",
    aboutContent: `
      <section class="space-y-6">
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">What is this?</h3>
          <p class="text-gray-700 dark:text-gray-300">
            This is a chatbot that can <strong>see and understand images</strong>. Upload a photo and ask questions about it: "What's in this picture?", "Can you read the text in this image?", or "What breed is this dog?" These capabilities are now built into services like ChatGPT, Claude, and Gemini when you share an image with them.
          </p>
        </div>
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">Going Deeper</h3>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            Image chat works differently from text-only conversation. Before the language model can process your question, an <strong>image encoder</strong> (often called a vision encoder) must first convert the image into a sequence of tokens the model can understand. A single image can become hundreds or thousands of tokens.
          </p>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            This has two implications for energy consumption. First, the <strong>input context is much longer</strong> than text-only chat, even for simple questions. Second, the <strong>vision encoder itself consumes energy</strong>, and for some models, the vision encoder is surprisingly large and computationally expensive.
          </p>
          <p class="text-gray-700 dark:text-gray-300">
            Vision encoder computation presents an interesting optimization opportunity: the same image encoding can potentially be <strong>reused across multiple questions</strong> about the same image. <a href="https://cornserve.ai" target="_blank" rel="noopener noreferrer" class="text-blue-600 dark:text-blue-400 hover:underline">Cornserve</a> explores this direction for efficient multimodal serving.
          </p>
        </div>
      </section>
    `,
  },
  "video-chat": {
    displayName: "Video Understanding and Conversation",
    tabLabel: "Video Chat",
    architecture: "mllm",
    defaultItlDeadlineMs: 500,
    defaultEnergyBudgetJ: null,
    description: "Multimodal video understanding and chat",
    aboutContent: `
      <section class="space-y-6">
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">What is this?</h3>
          <p class="text-gray-700 dark:text-gray-300">
            This is a chatbot that can <strong>watch and understand videos</strong>. Share a video clip and ask questions about it: "What happened in this video?", "Summarize the key points from this lecture", or "What is this person doing wrong in their golf swing?" Video understanding is one of the newest frontiers in AI assistants.
          </p>
        </div>
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">Going Deeper</h3>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            Video understanding scales up the challenges of image chat. Instead of encoding one image, the model must process <strong>many frames</strong>, sometimes dozens or hundreds. Each frame becomes tokens, so the input context can grow enormous. The model must also understand <strong>temporal dynamics</strong>: what happened first, what came next, how things changed over time.
          </p>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            Energy consumption scales with the number of frames processed. More frames mean better temporal understanding but higher energy costs. This creates an important trade-off: how many frames do you need to answer the question well? Some questions ("What color is the car?") might need only a single frame, while others ("Describe the full sequence of events") require many.
          </p>
          <p class="text-gray-700 dark:text-gray-300">
            As with image chat, the vision encoder's computation can potentially be <strong>cached and reused</strong> across multiple questions about the same video. <a href="https://cornserve.ai" target="_blank" rel="noopener noreferrer" class="text-blue-600 dark:text-blue-400 hover:underline">Cornserve</a> explores efficient approaches for multimodal serving.
          </p>
        </div>
      </section>
    `,
  },
  "audio-chat": {
    displayName: "Audio Understanding and Conversation",
    tabLabel: "Audio Chat",
    architecture: "mllm",
    defaultItlDeadlineMs: 500,
    defaultEnergyBudgetJ: null,
    description: "Multimodal audio understanding and chat",
    aboutContent: `
      <section class="space-y-6">
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">What is this?</h3>
          <p class="text-gray-700 dark:text-gray-300">
            This would be a chatbot that can <strong>hear and understand audio</strong>: identify sounds, transcribe speech, or answer questions about audio content. Audio understanding is an emerging capability for AI assistants.
          </p>
        </div>
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">Status</h3>
          <p class="text-gray-700 dark:text-gray-300">
            <strong>Coming soon.</strong> Audio chat benchmarks are not yet included in this version of the leaderboard.
          </p>
        </div>
      </section>
    `,
  },
  // Diffusion tasks
  "text-to-image": {
    displayName: "Image Generation from Text Prompts",
    tabLabel: "Text to Image",
    architecture: "diffusion",
    defaultLatencyDeadlineS: 10,
    defaultEnergyBudgetJ: null,
    description: "Generate images from text prompts",
    aboutContent: `
      <section class="space-y-6">
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">What is this?</h3>
          <p class="text-gray-700 dark:text-gray-300">
            This is <strong>AI image generation</strong>: describe what you want to see, and the AI creates it. "A cat wearing a spacesuit on Mars" or "An oil painting of a sunset over mountains." These are the capabilities behind image generation services that turn text prompts into artwork, product mockups, or creative content.
          </p>
        </div>
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">Going Deeper</h3>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            Image generation uses <strong>diffusion models</strong>, which work very differently from language models. Instead of generating tokens one by one, diffusion models start with random noise and gradually refine it into a coherent image through many <strong>denoising steps</strong>.
          </p>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            This means energy consumption scales <strong>linearly with the number of denoising steps</strong>. More steps generally mean higher quality but proportionally more time and energy. <strong>Image resolution</strong> also matters significantly: generating a 2K image requires far more computation than a 512×512 image.
          </p>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            The energy numbers shown here use <strong>model-recommended default settings</strong> (steps, resolution, guidance scale, etc.). These provide a fair comparison across models, but real-world usage often involves adjusting these parameters.
          </p>
          <p class="text-gray-700 dark:text-gray-300">
            <strong>Coming soon:</strong> We're preparing a control benchmark that holds the model constant while sweeping generation parameters, showing exactly how each setting affects energy consumption.
          </p>
        </div>
      </section>
    `,
  },
  "text-to-video": {
    displayName: "Video Generation from Text Prompts",
    tabLabel: "Text to Video",
    architecture: "diffusion",
    defaultLatencyDeadlineS: 120,
    defaultEnergyBudgetJ: null,
    description: "Generate videos from text prompts",
    aboutContent: `
      <section class="space-y-6">
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">What is this?</h3>
          <p class="text-gray-700 dark:text-gray-300">
            This is <strong>AI video generation</strong>: describe a scene, and the AI creates a video of it. "A drone flying over a coastal city at sunset" or "A timelapse of a flower blooming." Video generation is one of the most impressive and compute-intensive capabilities of modern AI.
          </p>
        </div>
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">Going Deeper</h3>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            Video generation extends image diffusion to the temporal dimension. The model must not only create visually coherent frames but also ensure <strong>smooth motion and temporal consistency</strong>: objects should move naturally, lighting should remain consistent, and the scene should evolve believably.
          </p>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            Energy consumption depends on multiple factors: <strong>number of frames</strong> (a 5-second video at 30fps needs 150 frames), <strong>resolution</strong> (4K vs 720p), and <strong>denoising steps</strong>. Each of these scales energy consumption roughly linearly. A high-resolution, long video with many denoising steps can consume orders of magnitude more energy than a short, low-resolution clip.
          </p>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            The energy numbers shown here use <strong>model-recommended default settings</strong>. These typically represent the model's intended quality/speed balance, but real deployments often customize these extensively.
          </p>
          <p class="text-gray-700 dark:text-gray-300">
            <strong>Coming soon:</strong> We're preparing a control benchmark that holds the model constant while sweeping generation parameters (resolution, frame count, steps), showing exactly how each setting affects energy consumption.
          </p>
        </div>
      </section>
    `,
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
