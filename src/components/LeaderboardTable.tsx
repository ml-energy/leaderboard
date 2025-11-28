import { useState, useEffect } from 'react';
import { AnyConfiguration, Configuration, ImageConfiguration, VideoConfiguration } from '../types';
import { ColumnDef } from '../config/columns';

interface LeaderboardTableProps {
  configurations: AnyConfiguration[];
  columns: ColumnDef[];
  defaultSortKey?: string;
  onRowClick?: (config: AnyConfiguration) => void;
  selectedForCompare?: Set<string>;
  onSelectionChange?: (modelId: string, selected: boolean) => void;
  onCompareClick?: () => void;
  onClearSelection?: () => void;
}

type SortDirection = 'asc' | 'desc' | null;

export default function LeaderboardTable({
  configurations,
  columns,
  defaultSortKey,
  onRowClick,
  selectedForCompare = new Set(),
  onSelectionChange,
  onCompareClick,
  onClearSelection,
}: LeaderboardTableProps) {
  const [sortKey, setSortKey] = useState<string | null>(defaultSortKey ?? null);
  const [sortDirection, setSortDirection] = useState<SortDirection>(defaultSortKey ? 'asc' : null);

  // Reset sort when columns or default sort key changes (task switch)
  useEffect(() => {
    setSortKey(defaultSortKey ?? null);
    setSortDirection(defaultSortKey ? 'asc' : null);
  }, [defaultSortKey]);

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

  const getCellValue = (config: AnyConfiguration, col: ColumnDef) => {
    const value = col.getValue ? col.getValue(config) : (config as any)[col.key];

    if (value === null || value === undefined) {
      return '-';
    }

    if (col.format) {
      return col.format(value);
    }

    return String(value);
  };

  // Generate unique row key for any configuration type
  const getRowKey = (config: AnyConfiguration): string => {
    const base = `${config.model_id}-${config.gpu_model}-${config.num_gpus}`;
    if ('batch_size' in config) {
      // Diffusion config - include parallelization for uniqueness
      const diffConfig = config as ImageConfiguration | VideoConfiguration;
      return `${base}-${diffConfig.batch_size}-${diffConfig.ulysses_degree}-${diffConfig.ring_degree}`;
    }
    // LLM/MLLM config
    const maxNumSeqs = 'max_num_seqs' in config ? (config as Configuration).max_num_seqs : null;
    return `${base}-${maxNumSeqs ?? 'null'}`;
  };

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
        <thead className="bg-gray-50 dark:bg-gray-800">
          <tr>
            {onSelectionChange && (
              <th className="w-40 min-w-40 px-2 py-2 border-r border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-center gap-1">
                  <button
                    onClick={onCompareClick}
                    disabled={selectedForCompare.size < 2}
                    className={`flex items-center gap-1 px-2 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap ${
                      selectedForCompare.size >= 2
                        ? 'text-white bg-blue-600 hover:bg-blue-700'
                        : 'text-gray-400 dark:text-gray-500 bg-gray-200 dark:bg-gray-700 cursor-not-allowed'
                    }`}
                    title={selectedForCompare.size >= 2 ? `Compare ${selectedForCompare.size} models` : 'Select 2+ models to compare'}
                  >
                    <svg className="w-3 h-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    {selectedForCompare.size >= 2 ? `Compare ${selectedForCompare.size}` : 'Compare'}
                  </button>
                  {selectedForCompare.size > 0 && onClearSelection && (
                    <button
                      onClick={onClearSelection}
                      className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                      title="Clear selection"
                    >
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  )}
                </div>
              </th>
            )}
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
              key={getRowKey(config)}
              onClick={() => onRowClick?.(config)}
              className={
                onRowClick
                  ? 'cursor-pointer hover:bg-blue-50 dark:hover:bg-gray-700 transition-colors'
                  : ''
              }
            >
              {onSelectionChange && (
                <td
                  className="w-40 min-w-40 px-3 py-4 border-r border-gray-200 dark:border-gray-700 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectionChange(config.model_id, !selectedForCompare.has(config.model_id));
                  }}
                >
                  <div className="flex justify-center">
                    <input
                      type="checkbox"
                      checked={selectedForCompare.has(config.model_id)}
                      onChange={(e) => {
                        e.stopPropagation();
                        onSelectionChange(config.model_id, e.target.checked);
                      }}
                      onClick={(e) => e.stopPropagation()}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded cursor-pointer"
                    />
                  </div>
                </td>
              )}
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
