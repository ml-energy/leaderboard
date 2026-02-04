export interface AnnouncementLink {
  text: string;
  url: string;
}

export interface Announcement {
  date: string;  // ISO date string (YYYY-MM-DD) - also serves as unique identifier
  message: string;
  links?: AnnouncementLink[];
}

// Announcements ordered from newest to oldest
export const announcements: Announcement[] = [
  {
    date: "2026-01-29",
    message: "New blog post: Diagnosing Inference Energy Consumption with the MLenergy Leaderboard v3.0",
    links: [
      { text: "Blog", url: "https://ml.energy/blog/measurement/energy/diagnosing-inference-energy-consumption-with-the-mlenergy-leaderboard-v30/" },
      { text: "arXiv", url: "https://arxiv.org/abs/2601.22076" },
    ],
  },
  {
    date: "2025-12-01",
    message: "Version 3.0 released! Epanded coverage of models, tasks, and hardware, with a complete new web UI.",
    links: [
      { text: "GitHub (Leaderboard)", url: "https://github.com/ml-energy/leaderboard" },
      { text: "GitHub (Benchmark)", url: "https://github.com/ml-energy/benchmark" },
    ],
  },
  {
    date: "2025-09-18",
    message: "Benchmark paper accepted at NeurIPS 25 D&B track as Spotlight paper!",
    links: [
      { text: "arXiv", url: "https://arxiv.org/abs/2505.06371" },
    ],
  },
  {
    date: "2024-09-19",
    message: "Version 2.0 released!",
    links: [
      { text: "GitHub", url: "https://github.com/ml-energy/leaderboard-v2" },
    ],
  },
  {
    date: "2023-07-06",
    message: "Version 1.0 released!",
    links: [
      { text: "GitHub", url: "https://github.com/ml-energy/leaderboard-v2" },
    ],
  },
];
