import * as React from "react"
import { cn } from "../../lib/utils"
import { X } from "lucide-react"
import { Button } from "./button"

const Dialog = ({ open, onOpenChange, children }) => {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black/50 backdrop-blur-sm"
        onClick={() => onOpenChange(false)}
      />
      
      {/* Content */}
      <div className="relative z-50 max-h-[90vh] overflow-auto">
        {children}
      </div>
    </div>
  )
}

const DialogContent = React.forwardRef(({ className, children, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "relative w-full max-w-lg rounded-lg border bg-background p-6 shadow-lg sm:max-w-2xl md:max-w-3xl",
      "animate-fade-in",
      className
    )}
    {...props}
  >
    {children}
  </div>
))
DialogContent.displayName = "DialogContent"

const DialogHeader = ({
  className,
  children,
  onClose,
  ...props
}) => (
  <div
    className={cn(
      "flex items-center justify-between space-y-1.5 pb-4 border-b",
      className
    )}
    {...props}
  >
    <div className="flex-1">
      {children}
    </div>
    {onClose && (
      <Button
        variant="ghost"
        size="icon"
        onClick={onClose}
        className="h-8 w-8"
      >
        <X className="h-4 w-4" />
      </Button>
    )}
  </div>
)
DialogHeader.displayName = "DialogHeader"

const DialogTitle = React.forwardRef(({ className, ...props }, ref) => (
  <h2
    ref={ref}
    className={cn(
      "text-lg font-semibold leading-none tracking-tight",
      className
    )}
    {...props}
  />
))
DialogTitle.displayName = "DialogTitle"

const DialogDescription = React.forwardRef(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn("text-sm text-muted-foreground", className)}
    {...props}
  />
))
DialogDescription.displayName = "DialogDescription"

const DialogBody = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("py-4", className)}
    {...props}
  />
))
DialogBody.displayName = "DialogBody"

const DialogFooter = ({
  className,
  ...props
}) => (
  <div
    className={cn(
      "flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2 pt-4 border-t",
      className
    )}
    {...props}
  />
)
DialogFooter.displayName = "DialogFooter"

export {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogBody,
  DialogFooter,
}