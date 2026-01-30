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
    <div className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white shadow-md">
      <div className="mx-auto py-2.5 px-6 flex items-center justify-center gap-3">
        <span className="text-sm">
          <span className="font-medium">{latestUnread.message}</span>
          {latestUnread.links && latestUnread.links.length > 0 && (
            <>
              {' '}
              {latestUnread.links.map((link, index) => (
                <span key={link.url}>
                  {index > 0 && ' · '}
                  <a
                    href={link.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 font-semibold underline decoration-2 underline-offset-2 hover:text-blue-100 transition-colors"
                  >
                    {link.text}
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                  </a>
                </span>
              ))}
            </>
          )}
        </span>
        <button
          onClick={() => onDismiss(latestUnread.date)}
          className="p-1 hover:bg-white/20 rounded transition-colors flex-shrink-0"
          aria-label="Dismiss announcement"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
}
