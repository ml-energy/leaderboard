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
          {/* Introduction */}
          <section>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              What is this?
            </h3>
            <p className="text-gray-700 dark:text-gray-300">
              The ML.ENERGY Leaderboard is a comprehensive benchmark for measuring the <strong>energy efficiency</strong> and <strong>performance</strong> of Large Language Models (LLMs), Multimodal LLMs (MLLMs), and diffusion models. We run real-world inference workloads using production-grade serving frameworks like vLLM while precisely measuring energy consumption with Zeus.
            </p>
          </section>

          {/* Methodology */}
          <section>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Methodology
            </h3>
            <div className="space-y-3 text-gray-700 dark:text-gray-300">
              <p>
                <strong>Benchmark Framework:</strong> We use vLLM as the inference engine, running inside Docker/Singularity containers on NVIDIA GPUs (H100, B200, etc.). Each model is tested with multiple batch sizes and GPU configurations to find the optimal energy-performance trade-off.
              </p>
              <p>
                <strong>Energy Measurement:</strong> Power consumption is measured at the GPU level using NVIDIA Management Library (NVML) through our Zeus energy measurement framework. We report steady-state energy per token (Joules/token) after the server reaches thermal and throughput equilibrium.
              </p>
              <p>
                <strong>Workloads:</strong> We use real-world datasets including:
              </p>
              <ul className="list-disc list-inside ml-4 space-y-1">
                <li><strong>LLM Chat (LM Arena):</strong> Conversational prompts from the LMSYS Arena dataset</li>
                <li><strong>GPQA Diamond:</strong> Graduate-level science questions requiring reasoning</li>
                <li><strong>Code Completion (Sourcegraph):</strong> Fill-in-the-middle code generation tasks</li>
                <li><strong>Image Chat:</strong> Vision-language tasks with image understanding</li>
                <li><strong>Video Chat:</strong> Video understanding and Q&A tasks</li>
              </ul>
            </div>
          </section>

          {/* Key Metrics */}
          <section>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Key Metrics Explained
            </h3>
            <div className="space-y-3">
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white">Energy/Token (J)</h4>
                <p className="text-gray-700 dark:text-gray-300">
                  Total energy consumed divided by the number of output tokens generated. Lower is better for energy efficiency.
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white">Median ITL (ms)</h4>
                <p className="text-gray-700 dark:text-gray-300">
                  Median Inter-Token Latency - the time between generating consecutive tokens. This measures the responsiveness and streaming quality of the model. We use ITL instead of Time-to-First-Token (TTFT) because our benchmarking methodology focuses on steady-state performance.
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white">Throughput (tok/s)</h4>
                <p className="text-gray-700 dark:text-gray-300">
                  Output tokens generated per second. Higher throughput means better hardware utilization and lower per-request costs.
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white">Total vs Active Parameters (MoE Models)</h4>
                <p className="text-gray-700 dark:text-gray-300">
                  For Mixture-of-Experts (MoE) models like DeepSeek-V3 or Qwen3-235B, we show both the total parameter count and the activated parameters per token. This helps understand the actual computational cost per inference.
                </p>
              </div>
            </div>
          </section>

          {/* How to Use */}
          <section>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              How to Use the Leaderboard
            </h3>
            <div className="space-y-3 text-gray-700 dark:text-gray-300">
              <p>
                <strong>1. Select a Task:</strong> Choose from LLM Chat, GPQA, Code Completion, Image Chat, or Video Chat based on your use case.
              </p>
              <p>
                <strong>2. Set Latency Deadline:</strong> Use the slider to filter models that meet your latency requirements. The leaderboard automatically shows the most energy-efficient configuration for each model that meets the deadline.
              </p>
              <p>
                <strong>3. Compare Models:</strong> Click on any row to see detailed performance metrics, distribution charts, and all tested configurations for that model.
              </p>
              <p>
                <strong>4. Filter by GPU:</strong> Toggle GPU filters to see results for specific hardware (H100, B200, etc.).
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
                <h4 className="font-semibold text-gray-900 dark:text-white">Why median ITL instead of TTFT?</h4>
                <p className="text-gray-700 dark:text-gray-300">
                  Our benchmarking methodology focuses on steady-state throughput with continuous request arrivals. In this setting, Inter-Token Latency (ITL) is more representative of the user experience than Time-to-First-Token (TTFT), which is heavily influenced by batching and scheduling rather than model performance.
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white">How are configurations selected?</h4>
                <p className="text-gray-700 dark:text-gray-300">
                  For each model and task, we sweep across multiple batch sizes and GPU counts. The leaderboard shows the configuration with the lowest energy per token that meets your latency deadline.
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white">What about FP8 quantization?</h4>
                <p className="text-gray-700 dark:text-gray-300">
                  Models with "-FP8" in their name use FP8 quantization, which reduces memory footprint and often improves energy efficiency. We benchmark both FP8 and full-precision (bfloat16) versions when available.
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white">Can I reproduce these results?</h4>
                <p className="text-gray-700 dark:text-gray-300">
                  Yes! All benchmark code, configurations, and data processing scripts are open source. Visit our GitHub repository for detailed instructions.
                </p>
              </div>
            </div>
          </section>

          {/* Links */}
          <section>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Links & Resources
            </h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>
                <strong>GitHub Repository:</strong>{' '}
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
                  href="mailto:contact@ml.energy"
                  className="text-blue-600 dark:text-blue-400 hover:underline"
                >
                  contact@ml.energy
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
              If you use this leaderboard in your research, please cite:
            </p>
            <pre className="bg-gray-100 dark:bg-gray-800 p-3 rounded text-xs overflow-x-auto">
{`@misc{mlenergy-leaderboard-v3,
  title={ML.ENERGY Leaderboard v3.0: Energy Efficiency Benchmark for LLMs and MLLMs},
  author={ML.ENERGY Initiative},
  year={2025},
  url={https://ml.energy/leaderboard}
}`}
            </pre>
          </section>
        </div>
      </div>
    </div>
  );
}
