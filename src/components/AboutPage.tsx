import { useEffect } from 'react';

export function AboutPage({ onClose }: { onClose: () => void }) {
  // Close on ESC key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4 flex justify-between items-center">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            About the ML.ENERGY Leaderboard
          </h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 text-2xl font-bold"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Badges */}
          <div className="flex flex-wrap gap-2">
            <a
              href="https://arxiv.org/abs/2505.06371"
              target="_blank"
              rel="noopener noreferrer"
            >
              <img
                src="https://img.shields.io/badge/NeurIPS'25_D&B-Spotlight-b31b1b?style=flat-square"
                alt="NeurIPS 2025 Datasets and Benchmarks Spotlight"
              />
            </a>
            <a
              href="https://github.com/ml-energy/benchmark"
              target="_blank"
              rel="noopener noreferrer"
            >
              <img
                src="https://img.shields.io/badge/ml--energy-benchmark-blue?style=flat-square&logo=github"
                alt="GitHub benchmark repo"
              />
            </a>
            <a
              href="https://github.com/ml-energy/leaderboard"
              target="_blank"
              rel="noopener noreferrer"
            >
              <img
                src="https://img.shields.io/badge/ml--energy-leaderboard-blue?style=flat-square&logo=github"
                alt="GitHub leaderboard repo"
              />
            </a>
          </div>

          {/* Introduction */}
          <section>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              What is this?
            </h3>
            <p className="text-gray-700 dark:text-gray-300">
              The ML.ENERGY Leaderboard visualizes the <strong>time-energy tradeoff</strong> of generative AI models, including Large Language Models (LLMs), Multimodal LLMs (MLLMs), and diffusion models. Our goal is to benchmark inference in <strong>realistic production-grade settings</strong> that reflect time and energy consumption <strong>at scale</strong>. See our <a href="https://neurips.cc/virtual/2025/loc/san-diego/poster/121781" target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-400 hover:underline">NeurIPS D&B paper</a> for details.
            </p>
          </section>

          {/* How to Use */}
          <section>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              How to Use the Leaderboard
            </h3>
            <div className="space-y-3 text-gray-700 dark:text-gray-300">
              <p>
                <strong>Browse the Whole Space:</strong> At the top of each task page, see the full range of benchmarked configurations for every model, including different batch sizes, GPU counts, and hardware options. You can <strong>hover oer points</strong> to see exact metrics and configurations. Also, clicking the legend item will open up a <strong>detailed model view</strong>.
              </p>
              <p>
                <strong>Set Constraints:</strong> Use the sliders to specify your latency requirement and energy budget. The leaderboard automatically filters and ranks models by their most energy-efficient configuration that meets your constraints. <strong>Each row on the table is clickable</strong>; click to open the detailed model view.
              </p>
              <p>
                <strong>Detailed Model View:</strong> Click on any table row or the big scatter plot's legend item to see all its tested configurations, the time-energy tradeoff curve, and all configurations benchmarked for that model.
              </p>
              <p>
                <strong>Compare Models:</strong> Select two or more models using the checkbox and compare their configurations side-by-side and see how they differ across the time-energy spectrum.
              </p>
            </div>
          </section>

          {/* Methodology */}
          <section>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Methodology
            </h3>
            <div className="space-y-3 text-gray-700 dark:text-gray-300">
              <p>
                <strong>Inference Engines:</strong> We use <a href="https://github.com/vllm-project/vllm" target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-400 hover:underline">vLLM</a> for LLMs and MLLMs, and <a href="https://github.com/xdit-project/xDiT" target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-400 hover:underline">xDiT</a> for diffusion models.
              </p>
              <p>
                <strong>Hardware:</strong> We ran our benchmark on NVIDIA H100 80GB HBM3 (max power 700 W per GPU) and NVIDIA B200 GPU (max power 1,000 W per GPU) servers.
              </p>
              <p>
                <strong>Energy Measurement:</strong> We measure and report <strong>GPU energy consumption</strong> using <a href="https://ml.energy/zeus" target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-400 hover:underline">Zeus</a>. We report steady-state GPU energy per output unit (Joules per token, image, or video) after the server reaches thermal and throughput equilibrium.
              </p>
              <p>
                <strong>Configuration Sweeping:</strong> For each model and task, we sweep across multiple batch sizes, GPU models, and number of GPUs to map out the time-energy tradeoff space. This allows users to find the optimal configuration for their specific latency and energy constraints.
              </p>
            </div>
          </section>

          {/* Workloads */}
          <section>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Workloads
            </h3>
            <div className="space-y-3 text-gray-700 dark:text-gray-300">
              <p>We benchmark models on realistic use cases:</p>
              <ul className="list-disc list-inside ml-4 space-y-1">
                <li><strong>Problem Solving:</strong> Working through math problems, debugging code, or explaining concepts step by step</li>
                <li><strong>Text Conversation:</strong> Chatting with an AI assistant to get help writing, brainstorm ideas, or answer questions</li>
                <li><strong>Code Completion:</strong> Inline "tab to accept" code suggestions in AI-powered editors</li>
                <li><strong>Image Chat:</strong> Uploading a photo and asking questions about it</li>
                <li><strong>Video Chat:</strong> Sharing a video clip and asking questions about it</li>
                <li><strong>Text to Image:</strong> Describing what you want to see and having the AI create an image</li>
                <li><strong>Text to Video:</strong> Describing a scene and having the AI create a video</li>
              </ul>
            </div>
          </section>

          {/* Key Metrics */}
          <section>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Key Metrics
            </h3>
            <div className="space-y-3 text-gray-700 dark:text-gray-300">
              <p>
                <strong>Response Time:</strong> How long it takes to generate a complete response. For LLMs and MLLMs, this is the time to produce the entire text response. For diffusion models, this is the time to generate the full image or video.
              </p>
              <p>
                <strong>Energy per Response:</strong> Total GPU energy consumed to generate one response. We report this at the granularity of individual output units (tokens for LLMs, images for text-to-image, etc.) to enable fair comparison across models with different output lengths.
              </p>
            </div>
          </section>

          {/* FAQ */}
          <section>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Frequently Asked Questions
            </h3>
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white">Is this accurate?</h4>
                <p className="text-gray-700 dark:text-gray-300">
                  We take accuracy seriously. Our measurements use real GPU energy readings (not max power-based estimates, which can lead to significant overestimations). We measure steady-state energy when the server is running at its configured batch size, capturing realistic deployment behavior. All benchmarks run on production-grade hardware and software stacks. See our <a href="https://arxiv.org/abs/2505.06371" target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-400 hover:underline">paper</a> for detailed methodology.
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white">What are the limitations?</h4>
                <p className="text-gray-700 dark:text-gray-300">
                  We can only benchmark <strong>open-weight models</strong>; closed models like OpenAI GPT or Claude cannot be measured. Software and hardware infrastructure evolves over time, so results may drift slightly from optimal as new versions are released. We also cannot cover every possible optimization (e.g., speculative decoding, prefill-decode disaggregation), though we include the most common and impactful ones and follow deployment best practices. However, we are planning to follow-up with a <strong>control benchmark</strong> that selects one or two important models and test out various workloads and optimizations.
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white">Can I reproduce these results?</h4>
                <p className="text-gray-700 dark:text-gray-300">
                  Yes! All benchmark code, configurations, and data processing scripts are open source. Visit our GitHub repositories for detailed instructions.
                </p>
              </div>
            </div>
          </section>

          {/* Version History */}
          <section>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Version History
            </h3>
            <div className="space-y-3 text-gray-700 dark:text-gray-300">
              <p>
                <strong>V3.0</strong> (December 1, 2025): Current version with LLMs, MLLMs, and diffusion models.{' '}
                <a
                  href="https://github.com/ml-energy/leaderboard"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 dark:text-blue-400 hover:underline"
                >
                  GitHub
                </a>
              </p>
              <p>
                <strong>V2.0</strong> (September 20, 2024): Added multi-GPU configurations and more models.{' '}
                <a
                  href="https://github.com/ml-energy/leaderboard-v2"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 dark:text-blue-400 hover:underline"
                >
                  GitHub
                </a>
              </p>
              <p>
                <strong>V1.0</strong> (July 6, 2023): Initial release with single-GPU LLM benchmarks.{' '}
                <a
                  href="https://github.com/ml-energy/leaderboard-v2/releases/tag/2023-07-06"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 dark:text-blue-400 hover:underline"
                >
                  GitHub
                </a>
              </p>
            </div>
          </section>

          {/* Links */}
          <section>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Links & Resources
            </h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>
                <strong>Benchmark Code:</strong>{' '}
                <a
                  href="https://github.com/ml-energy/benchmark"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 dark:text-blue-400 hover:underline"
                >
                  ml-energy/benchmark
                </a>
              </li>
              <li>
                <strong>Leaderboard Code:</strong>{' '}
                <a
                  href="https://github.com/ml-energy/leaderboard"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 dark:text-blue-400 hover:underline"
                >
                  ml-energy/leaderboard
                </a>
              </li>
              <li>
                <strong>Zeus Framework:</strong>{' '}
                <a
                  href="https://ml.energy/zeus"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 dark:text-blue-400 hover:underline"
                >
                  ml.energy/zeus
                </a>
              </li>
              <li>
                <strong>Contact:</strong>{' '}
                <a
                  href="mailto:admins@ml.energy"
                  className="text-blue-600 dark:text-blue-400 hover:underline"
                >
                  admins@ml.energy
                </a>
              </li>
            </ul>
          </section>

          {/* Citation */}
          <section className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              Citation
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              If you use this leaderboard in your research, please cite our NeurIPS 2025 D&B paper:
            </p>
            <pre className="bg-gray-100 dark:bg-gray-800 p-3 rounded text-xs overflow-x-auto text-gray-800 dark:text-gray-200">
{`@inproceedings{mlenergy-neuripsdb25,
    title={The {ML.ENERGY Benchmark}: Toward Automated Inference Energy Measurement and Optimization},
    author={Jae-Won Chung and Jeff J. Ma and Ruofan Wu and Jiachen Liu and Oh Jun Kweon and Yuxuan Xia and Zhiyu Wu and Mosharaf Chowdhury},
    year={2025},
    booktitle={NeurIPS Datasets and Benchmarks},
}`}
            </pre>
          </section>
        </div>
      </div>
    </div>
  );
}
