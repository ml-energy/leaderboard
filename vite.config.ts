import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { execSync } from 'child_process'

// Get the latest git commit date
const getGitCommitDate = () => {
  try {
    const date = execSync('git log -1 --format=%ci').toString().trim()
    // Format: "2025-12-01 10:30:00 -0500" -> "December 1, 2025"
    const d = new Date(date)
    return d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric', timeZone: 'America/New_York' })
  } catch {
    return 'Unknown'
  }
}

export default defineConfig({
  plugins: [react()],
  base: '/leaderboard/',
  define: {
    __LAST_UPDATED__: JSON.stringify(getGitCommitDate()),
  },
})
