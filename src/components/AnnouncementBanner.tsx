import { announcements } from '../config/announcements';

interface AnnouncementBannerProps {
  readIds: Set<string>;
  onDismiss: (id: string) => void;
}

export function AnnouncementBanner({ readIds, onDismiss }: AnnouncementBannerProps) {
  // Find the latest unread announcement (announcements are ordered newest first)
  const latestUnread = announcements.find(a => !readIds.has(a.date));

  if (!latestUnread) {
    return null;
  }

  return (
    <div className="bg-blue-600 text-white">
      <div className="mx-auto py-3 px-6 flex items-center justify-between">
        <div className="flex-1 text-center text-sm">
          <div>{latestUnread.message}</div>
          {latestUnread.links && latestUnread.links.length > 0 && (
            <div className="mt-1">
              {latestUnread.links.map((link, index) => (
                <span key={link.url}>
                  {index > 0 && ' · '}
                  <a
                    href={link.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline font-medium hover:text-blue-100"
                  >
                    {link.text}
                  </a>
                </span>
              ))}
            </div>
          )}
        </div>
        <button
          onClick={() => onDismiss(latestUnread.date)}
          className="ml-4 p-1 hover:bg-blue-700 rounded transition-colors"
          aria-label="Dismiss announcement"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
}
