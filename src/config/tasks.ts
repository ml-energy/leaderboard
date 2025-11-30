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
            This has noticeable implications for energy consumption. Long outputs mean the model's context grows throughout generation, which increases memory consumption and reduces how many responses the server can generate in parallel, or in other words, <strong>smaller batch size</strong>. When fewer requests can be processed together, the GPU's computational capacity is less efficiently utilized, making each token more expensive to generate. Combined with the sheer number of output tokens, <strong>energy per response becomes very large</strong> compared to shorter conversational tasks.
          </p>
          <p class="text-gray-700 dark:text-gray-300">
            This is why reasoning models like DeepSeek-R1 or Qwen3 with thinking mode show dramatically different energy profiles than the same base models in chat mode.
          </p>
        </div>
      </section>
    `,
  },
  "lm-arena-chat": {
    displayName: "Conversational AI Chatbot",
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
            Text conversation represents a large portion of daily LLM usage today. Both inputs and outputs are relatively short: a typical user message is a sentence or two, and responses are a few paragraphs at most.
          </p>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            One distinct aspect of conversational AI, be it text-based or audio-based, is that a human user is sitting in front of the service, reading or listening to the generated content. However, users do not read or listen at infinitely fast speeds; this creates a natural <strong>loose latency deadline</strong> for generating output tokens, and the server can increase batch size a lot to improve energy-efficiency without negatively impacting user experience. <a href="https://arxiv.org/abs/2404.16283" target="_blank" rel="noopener noreferrer" class="text-blue-600 dark:text-blue-400 hover:underline">This paper</a> explores user experience in more detail.
          </p>
          <p class="text-gray-700 dark:text-gray-300">
            Another difference is that the conversation history accumulates over <strong>multiple turns</strong>, so the model often sees a longer context of accumulated past conversations, rather than just the latest message. However, efficient <strong>prefix caching</strong> allows the server to avoid repeatedly processing the full conversation history for each request, allowing faster and more efficient response generation.
          </p>
        </div>
      </section>
    `,
  },
  "sourcegraph-fim": {
    displayName: "Fill-in-the-Middle Code Completion",
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
            This is the <strong>inline code suggestion</strong> you see when typing in a code editor. An LLM predicts what you're about to type and offers to complete it for you. Think of the "tab to accept" suggestions in AI-powered editors. The model sees the code before and after your cursor and fills in the gap.
          </p>
        </div>
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">Going Deeper</h3>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            Code completion has the <strong>opposite profile from problem-solving</strong>: inputs are long (the surrounding code context, and sometimes context about the entire codebase), but outputs are short (typically just a couple to tens of lines of code). The model needs substantial context to understand what code would fit, but the actual completion is brief, at least compared to problem solving with reasoning.
          </p>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            Fill-in-the-middle code completion is a well-known task, but we found that it's not widely supported by open-weight models and open-source LLM inference servers (models that claim support didn't work as well either), likely because not many people are serving their own complex code completion AI assistants yet. But, among recent models, Qwen 3 Coder worked well, which we currently benchmarked. We are looking forward to adding more models to this task in the future.
          </p>
        </div>
      </section>
    `,
  },
  // MLLM tasks
  "image-chat": {
    displayName: "Conversational AI Chatbot with Image Understanding",
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
            This has a few implications for the server's runtime. First, the <strong>input context is much longer</strong> than text-only chat, even for simple questions. Second, the <strong>vision encoder itself consumes time and energy</strong>, and for some models, the vision encoder is surprisingly large and computationally expensive. Finally, processing image pixels into a format that the vision encoder can handle can also take non-trivial CPU time, and this may sometimes become a bottleneck. The energy implication of this is that the server runs with a <strong>lower batch size</strong>, leading to less efficient energy amortization and higher energy per token & response.
          </p>
          <p class="text-gray-700 dark:text-gray-300">
            Vision encoder computation presents an interesting optimization opportunity: the vision encoder (or, really any subcomponent of the model like audio generators) can be disaggregated out of the monolithic serving system and served separately to reduce interference and server code complexity. <a href="https://cornserve.ai" target="_blank" rel="noopener noreferrer" class="text-blue-600 dark:text-blue-400 hover:underline">Cornserve</a> explores this direction for efficient multimodal serving.
          </p>
        </div>
      </section>
    `,
  },
  "video-chat": {
    displayName: "Conversational AI Chatbot with Video Understanding",
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
            This is a chatbot that can <strong>watch and understand videos</strong>. Share a video clip and ask questions about it: "What happened in this video?", "Summarize the key points from this lecture", or "What is this person doing wrong in their golf swing?"
          </p>
        </div>
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">Going Deeper</h3>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            Video chat scales up the challenges of image chat. Instead of encoding one image, an <strong>image encoder</strong> (often called a vision encoder) must process <strong>many frames</strong>, sometimes dozens. Each frame becomes tokens, so the input context can grow enormous. To some degree this is inevitable, as the model must understand <strong>temporal dynamics</strong>: what happened first, what came next, how things changed over time.
          </p>
          <p class="text-gray-700 dark:text-gray-300 mb-3">
            This has significant implications for the server's runtime. First, the <strong>input context is much longer</strong> than image chat, scaling with the number of frames. Second, the <strong>vision encoder's computation volume increases significantly</strong>, multiplying its time and energy consumption. Finally, processing video frames into a format the vision encoder can handle takes substantial CPU time, which can become a bottleneck. The energy implication is that the server runs with an <strong>even lower batch size</strong> than image chat, leading to less efficient energy amortization and higher energy per token & response.
          </p>
          <p class="text-gray-700 dark:text-gray-300">
            As with image chat, the vision encoder can be disaggregated out of the monolithic serving system and served separately to reduce interference and server code complexity. <a href="https://cornserve.ai" target="_blank" rel="noopener noreferrer" class="text-blue-600 dark:text-blue-400 hover:underline">Cornserve</a> explores this direction for efficient multimodal serving.
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
    defaultLatencyDeadlineS: 30,
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
            This means energy consumption scales <strong>almost linearly with the number of denoising steps</strong>. More steps generally mean higher quality (with diminishing returns) but proportionally more time and energy. <strong>Image resolution</strong> also matters significantly: generating a 2K image requires far more computation than a 512×512 image.
          </p>
          <p class="text-gray-700 dark:text-gray-300">
            <strong>Coming soon:</strong> The energy numbers shown here use <strong>model-recommended default settings</strong> (steps, resolution, etc.). But real-world usage often involves adjusting these parameters. We're preparing a control benchmark that holds the model constant while sweeping generation parameters, showing exactly how each setting affects energy consumption.
          </p>
        </div>
      </section>
    `,
  },
  "text-to-video": {
    displayName: "Video Generation from Text Prompts",
    tabLabel: "Text to Video",
    architecture: "diffusion",
    defaultLatencyDeadlineS: 300,
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
            Energy consumption depends on multiple factors: <strong>number of frames</strong> (a 5-second video at 30fps needs 150 frames), <strong>resolution</strong> (4K vs 720p), and <strong>denoising steps</strong>. All of these dimensions influence time and energy consumption significantly. A high-resolution, long video with many denoising steps can consume orders of magnitude more energy than a short, low-resolution clip.
          </p>
          <p class="text-gray-700 dark:text-gray-300">
            <strong>Coming soon:</strong> The energy numbers shown here use <strong>model-recommended default settings</strong> (steps, resolution, number of frames, etc.). But real-world usage often involves adjusting these parameters. We're preparing a control benchmark that holds the model constant while sweeping generation parameters, showing exactly how each setting affects energy consumption.
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
