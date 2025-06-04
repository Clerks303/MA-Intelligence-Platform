import React, { useState, useMemo } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from './table';
import { Button } from './button';
import { Input } from './input';
import { 
  ChevronLeft, 
  ChevronRight, 
  ChevronsLeft, 
  ChevronsRight,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Search
} from 'lucide-react';
import { cn } from '../../lib/utils';

export function DataTable({
  data = [],
  columns = [],
  searchable = true,
  searchPlaceholder = "Rechercher...",
  pageSize = 10,
  className,
  onRowClick,
  loading = false
}) {
  const [sorting, setSorting] = useState({ key: null, direction: 'asc' });
  const [currentPage, setCurrentPage] = useState(1);
  const [globalFilter, setGlobalFilter] = useState('');

  // Filter data based on global search
  const filteredData = useMemo(() => {
    if (!globalFilter) return data;
    
    return data.filter(row =>
      columns.some(column => {
        const value = row[column.accessorKey];
        return value && value.toString().toLowerCase().includes(globalFilter.toLowerCase());
      })
    );
  }, [data, globalFilter, columns]);

  // Sort data
  const sortedData = useMemo(() => {
    if (!sorting.key) return filteredData;
    
    return [...filteredData].sort((a, b) => {
      const aValue = a[sorting.key];
      const bValue = b[sorting.key];
      
      if (aValue === null || aValue === undefined) return 1;
      if (bValue === null || bValue === undefined) return -1;
      
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sorting.direction === 'asc' ? aValue - bValue : bValue - aValue;
      }
      
      const aStr = aValue.toString().toLowerCase();
      const bStr = bValue.toString().toLowerCase();
      
      if (sorting.direction === 'asc') {
        return aStr.localeCompare(bStr);
      } else {
        return bStr.localeCompare(aStr);
      }
    });
  }, [filteredData, sorting]);

  // Paginate data
  const totalPages = Math.ceil(sortedData.length / pageSize);
  const startIndex = (currentPage - 1) * pageSize;
  const paginatedData = sortedData.slice(startIndex, startIndex + pageSize);

  const handleSort = (key) => {
    setSorting(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  const getSortIcon = (key) => {
    if (sorting.key !== key) {
      return <ArrowUpDown className="ml-2 h-4 w-4" />;
    }
    return sorting.direction === 'asc' 
      ? <ArrowUp className="ml-2 h-4 w-4" />
      : <ArrowDown className="ml-2 h-4 w-4" />;
  };

  const handlePageChange = (page) => {
    setCurrentPage(Math.max(1, Math.min(page, totalPages)));
  };

  if (loading) {
    return (
      <div className={cn("space-y-4", className)}>
        {searchable && (
          <div className="relative w-full max-w-sm">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder={searchPlaceholder}
              className="pl-10"
              disabled
            />
          </div>
        )}
        <div className="rounded-md border">
          <div className="h-64 flex items-center justify-center">
            <div className="flex items-center space-x-2">
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              <span className="text-muted-foreground">Chargement...</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("space-y-4", className)}>
      {/* Search */}
      {searchable && (
        <div className="relative w-full max-w-sm">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder={searchPlaceholder}
            value={globalFilter}
            onChange={(e) => {
              setGlobalFilter(e.target.value);
              setCurrentPage(1);
            }}
            className="pl-10"
          />
        </div>
      )}

      {/* Table */}
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              {columns.map((column) => (
                <TableHead
                  key={column.accessorKey}
                  className={cn(
                    column.sortable !== false && "cursor-pointer select-none hover:bg-muted/50",
                    column.headerClassName
                  )}
                  style={{ width: column.width }}
                  onClick={() => column.sortable !== false && handleSort(column.accessorKey)}
                >
                  <div className="flex items-center">
                    {column.header}
                    {column.sortable !== false && getSortIcon(column.accessorKey)}
                  </div>
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {paginatedData.length === 0 ? (
              <TableRow>
                <TableCell colSpan={columns.length} className="h-24 text-center">
                  Aucun résultat trouvé.
                </TableCell>
              </TableRow>
            ) : (
              paginatedData.map((row, index) => (
                <TableRow
                  key={row.id || index}
                  className={cn(
                    onRowClick && "cursor-pointer",
                    "hover:bg-muted/50 transition-colors"
                  )}
                  onClick={() => onRowClick?.(row)}
                >
                  {columns.map((column) => (
                    <TableCell
                      key={column.accessorKey}
                      className={column.cellClassName}
                    >
                      {column.cell 
                        ? column.cell({ getValue: () => row[column.accessorKey], row })
                        : row[column.accessorKey]
                      }
                    </TableCell>
                  ))}
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <div className="text-sm text-muted-foreground">
            Affichage de {startIndex + 1} à {Math.min(startIndex + pageSize, sortedData.length)} sur {sortedData.length} résultats
          </div>
          
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => handlePageChange(1)}
              disabled={currentPage === 1}
            >
              <ChevronsLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => handlePageChange(currentPage - 1)}
              disabled={currentPage === 1}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            
            <div className="flex items-center space-x-1">
              <span className="text-sm text-muted-foreground">Page</span>
              <span className="text-sm font-medium">
                {currentPage} sur {totalPages}
              </span>
            </div>
            
            <Button
              variant="outline"
              size="sm"
              onClick={() => handlePageChange(currentPage + 1)}
              disabled={currentPage === totalPages}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => handlePageChange(totalPages)}
              disabled={currentPage === totalPages}
            >
              <ChevronsRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}