import { useState } from 'react';
import { Configuration } from '../types';
import { ColumnDef } from '../config/columns';

interface LeaderboardTableProps {
  configurations: Configuration[];
  columns: ColumnDef[];
  onRowClick?: (config: Configuration) => void;
}

type SortDirection = 'asc' | 'desc' | null;

export default function LeaderboardTable({
  configurations,
  columns,
  onRowClick
}: LeaderboardTableProps) {
  const [sortKey, setSortKey] = useState<string | null>('energy_per_token_joules');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');

  const handleSort = (key: string, sortable: boolean) => {
    if (!sortable) return;

    if (sortKey === key) {
      if (sortDirection === 'asc') {
        setSortDirection('desc');
      } else if (sortDirection === 'desc') {
        setSortDirection(null);
        setSortKey(null);
      }
    } else {
      setSortKey(key);
      setSortDirection('asc');
    }
  };

  const sortedConfigs = [...configurations].sort((a, b) => {
    if (!sortKey || !sortDirection) return 0;

    const col = columns.find(c => c.key === sortKey);
    const aVal = col?.getValue ? col.getValue(a) : (a as any)[sortKey];
    const bVal = col?.getValue ? col.getValue(b) : (b as any)[sortKey];

    if (aVal === null || aVal === undefined) return 1;
    if (bVal === null || bVal === undefined) return -1;

    if (typeof aVal === 'string' && typeof bVal === 'string') {
      return sortDirection === 'asc'
        ? aVal.localeCompare(bVal)
        : bVal.localeCompare(aVal);
    }

    return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
  });

  const getSortIcon = (key: string, sortable: boolean) => {
    if (!sortable) return null;

    if (sortKey !== key) {
      return <span className="ml-1 text-gray-400">⇅</span>;
    }

    if (sortDirection === 'asc') {
      return <span className="ml-1 text-blue-600">↑</span>;
    } else if (sortDirection === 'desc') {
      return <span className="ml-1 text-blue-600">↓</span>;
    }

    return <span className="ml-1 text-gray-400">⇅</span>;
  };

  const getCellValue = (config: Configuration, col: ColumnDef) => {
    const value = col.getValue ? col.getValue(config) : (config as any)[col.key];

    if (value === null || value === undefined) {
      return '-';
    }

    if (col.format) {
      return col.format(value);
    }

    return String(value);
  };

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
        <thead className="bg-gray-50 dark:bg-gray-800">
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                onClick={() => handleSort(col.key, col.sortable)}
                className={`px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 ${
                  col.sortable ? 'cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700' : ''
                }`}
              >
                <div className="flex items-center">
                  {col.label}
                  {getSortIcon(col.key, col.sortable)}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
          {sortedConfigs.map((config) => (
            <tr
              key={`${config.model_id}-${config.gpu_model}-${config.num_gpus}-${config.max_num_seqs}`}
              onClick={() => onRowClick?.(config)}
              className={`${
                onRowClick
                  ? 'cursor-pointer hover:bg-blue-50 dark:hover:bg-gray-700 transition-colors'
                  : ''
              }`}
            >
              {columns.map((col) => (
                <td
                  key={col.key}
                  className="px-3 py-4 whitespace-nowrap text-base text-gray-900 dark:text-gray-100"
                >
                  {getCellValue(config, col)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {sortedConfigs.length === 0 && (
        <div className="text-center py-12 text-gray-500 dark:text-gray-400">
          No configurations match the current filters.
        </div>
      )}
    </div>
  );
}
