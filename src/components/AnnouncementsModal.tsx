import { announcements, Announcement } from '../config/announcements';

interface AnnouncementsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function AnnouncementsModal({
  isOpen,
  onClose,
}: AnnouncementsModalProps) {
  if (!isOpen) return null;

  const formatDate = (dateStr: string) => {
    const [year, month, day] = dateStr.split('-').map(Number);
    const months = [
      'January', 'February', 'March', 'April', 'May', 'June',
      'July', 'August', 'September', 'October', 'November', 'December'
    ];
    return `${months[month - 1]} ${day}, ${year}`;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/50"
        onClick={onClose}
      />
      <div className="relative bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-lg w-full mx-4 max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            Announcements
          </h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
            aria-label="Close"
          >
            <svg className="w-5 h-5 text-gray-500 dark:text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {announcements.length === 0 ? (
            <p className="text-gray-500 dark:text-gray-400 text-center py-8">
              No announcements yet.
            </p>
          ) : (
            announcements.map((announcement: Announcement) => (
              <div
                key={announcement.date}
                className="p-4 rounded-lg border bg-gray-50 dark:bg-gray-900 border-gray-200 dark:border-gray-700"
              >
                <div className="text-sm text-gray-500 dark:text-gray-400 mb-1">
                  {formatDate(announcement.date)}
                </div>
                <p className="text-gray-900 dark:text-white">
                  {announcement.message}
                </p>
                {announcement.links && announcement.links.length > 0 && (
                  <div className="mt-1">
                    {announcement.links.map((link, index) => (
                      <span key={link.url}>
                        {index > 0 && ' · '}
                        <a
                          href={link.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 dark:text-blue-400 hover:underline"
                        >
                          {link.text}
                        </a>
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
