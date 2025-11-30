interface HeaderProps {
  darkMode: boolean;
  onDarkModeToggle: () => void;
  onAboutClick: () => void;
}

export function Header({ darkMode, onDarkModeToggle, onAboutClick }: HeaderProps) {
  return (
    <header className="relative overflow-hidden bg-gray-950 border-b border-gray-800">
      {/* Grid pattern background */}
      <div
        className="absolute inset-0 opacity-[0.04]"
        style={{
          backgroundImage: `
            linear-gradient(to right, #ffffff 1px, transparent 1px),
            linear-gradient(to bottom, #ffffff 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px',
        }}
      />

      {/* Subtle gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-r from-gray-950 via-transparent to-gray-950" />

      {/* Scan line effect */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.02]"
        style={{
          backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.03) 2px, rgba(255,255,255,0.03) 4px)',
        }}
      />

      <div className="relative mx-auto py-8 px-6 flex justify-between items-center">
        <div className="flex items-center gap-6">
          {/* Logo/Brand */}
          <div className="flex flex-col">
            <div className="flex items-baseline">
              <span
                className="text-4xl font-extrabold tracking-tight text-white"
                style={{ fontFamily: "'Montserrat', sans-serif" }}
              >
                ML.ENERGY
              </span>
            </div>
            <div className="flex items-center gap-3 mt-2">
              <div className="h-px flex-1 bg-gradient-to-r from-white/30 to-transparent" />
              <span
                className="text-sm uppercase tracking-[0.2em] text-gray-400 font-medium"
                style={{ fontFamily: "'Montserrat', sans-serif" }}
              >
                Leaderboard
              </span>
              <div className="h-px flex-1 bg-gradient-to-l from-white/30 to-transparent" />
            </div>
          </div>

          {/* Separator */}
          <div className="hidden md:block h-12 w-px bg-gray-800" />

          {/* Tagline */}
          <p
            className="hidden md:block text-sm text-gray-400 max-w-xs leading-relaxed"
            style={{ fontFamily: "'Instrument Sans', sans-serif" }}
          >
            Benchmarking time and energy consumption of generative AI models
          </p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-3">
          {/* Dark mode toggle */}
          <button
            onClick={onDarkModeToggle}
            className="p-2.5 rounded-lg bg-gray-900 border border-gray-800 hover:border-gray-600 hover:bg-gray-800 transition-all duration-200"
            title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {darkMode ? (
              <svg className="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" />
              </svg>
            ) : (
              <svg className="w-5 h-5 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
                <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
              </svg>
            )}
          </button>

          {/* About button */}
          <button
            onClick={onAboutClick}
            className="px-4 py-2.5 bg-white hover:bg-gray-200 text-gray-950 font-semibold rounded-lg transition-all duration-200 text-sm"
            style={{ fontFamily: "'Montserrat', sans-serif" }}
          >
            About
          </button>
        </div>
      </div>

      {/* Bottom accent line */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
    </header>
  );
}
