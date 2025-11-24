interface TaskTabsProps {
  tasks: string[];
  activeTask: string;
  onTaskChange: (task: string) => void;
}

const TASK_DISPLAY_NAMES: Record<string, string> = {
  'lm-arena-chat': 'LLM Chat',
  'gpqa': 'GPQA',
  'sourcegraph-fim': 'Code Completion',
  'image-chat': 'Image Chat',
  'video-chat': 'Video Chat',
  'audio-chat': 'Audio Chat',
};

export default function TaskTabs({ tasks, activeTask, onTaskChange }: TaskTabsProps) {
  const getDisplayName = (task: string) => {
    return TASK_DISPLAY_NAMES[task] || task;
  };

  return (
    <div className="border-b border-gray-200 dark:border-gray-700">
      <nav className="-mb-px flex space-x-8 overflow-x-auto" aria-label="Tabs">
        {tasks.map((task) => (
          <button
            key={task}
            onClick={() => onTaskChange(task)}
            className={`
              whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition-colors
              ${
                activeTask === task
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
              }
            `}
            aria-current={activeTask === task ? 'page' : undefined}
          >
            {getDisplayName(task)}
          </button>
        ))}
      </nav>
    </div>
  );
}
