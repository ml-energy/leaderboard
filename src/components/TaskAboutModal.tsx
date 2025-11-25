import { getTaskConfig } from '../config/tasks';

interface TaskAboutModalProps {
  taskId: string;
  onClose: () => void;
}

export function TaskAboutModal({ taskId, onClose }: TaskAboutModalProps) {
  const config = getTaskConfig(taskId);

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4 flex justify-between items-center">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            About: {config.displayName}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 text-2xl font-bold"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {config.aboutContent ? (
            <div
              className="prose dark:prose-invert max-w-none"
              dangerouslySetInnerHTML={{ __html: config.aboutContent }}
            />
          ) : (
            <div className="text-gray-600 dark:text-gray-400">
              <p className="mb-4">
                {config.description || `Benchmark results for ${config.displayName}.`}
              </p>
              <p className="text-sm italic">
                Detailed documentation for this task has not been added yet.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
