import * as React from "react"

import { cn } from "../../lib/utils"

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  icon?: React.ReactNode
  error?: string
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, icon, error, ...props }, ref) => {
    return (
      <div className="relative">
        {icon && (
          <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground">
            {icon}
          </div>
        )}
        <input
          type={type}
          className={cn(
            "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
            icon && "pl-10",
            error && "border-ma-red-500 focus-visible:ring-ma-red-500",
            className
          )}
          ref={ref}
          {...props}
        />
        {error && (
          <p className="mt-1 text-xs text-ma-red-600">{error}</p>
        )}
      </div>
    )
  }
)
Input.displayName = "Input"

// Search Input variant for M&A Intelligence
const SearchInput = React.forwardRef<
  HTMLInputElement,
  InputProps & { 
    onClear?: () => void
    showClear?: boolean
  }
>(({ className, onClear, showClear, ...props }, ref) => {
  return (
    <div className="relative">
      <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground">
        <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
      </div>
      <Input
        ref={ref}
        className={cn("pl-10", showClear && "pr-10", className)}
        {...props}
      />
      {showClear && props.value && onClear && (
        <button
          type="button"
          onClick={onClear}
          className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
        >
          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </div>
  )
})
SearchInput.displayName = "SearchInput"

export { Input, SearchInput }