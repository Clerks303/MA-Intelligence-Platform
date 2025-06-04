import * as React from "react"
import { cn } from "../../lib/utils"
import { AlertCircle, CheckCircle, Info, AlertTriangle } from "lucide-react"

const alertVariants = {
  default: "bg-background text-foreground",
  destructive:
    "border-destructive/50 text-destructive dark:border-destructive [&>svg]:text-destructive",
  success:
    "border-green-500/50 text-green-700 bg-green-50 dark:border-green-500/50 dark:text-green-400 dark:bg-green-950 [&>svg]:text-green-500",
  warning:
    "border-yellow-500/50 text-yellow-700 bg-yellow-50 dark:border-yellow-500/50 dark:text-yellow-400 dark:bg-yellow-950 [&>svg]:text-yellow-500",
  info:
    "border-blue-500/50 text-blue-700 bg-blue-50 dark:border-blue-500/50 dark:text-blue-400 dark:bg-blue-950 [&>svg]:text-blue-500",
}

const Alert = React.forwardRef(({ className, variant = "default", ...props }, ref) => (
  <div
    ref={ref}
    role="alert"
    className={cn(
      "relative w-full rounded-lg border p-4 [&>svg~*]:pl-7 [&>svg+div]:translate-y-[-3px] [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-4 [&>svg]:text-foreground",
      alertVariants[variant],
      className
    )}
    {...props}
  />
))
Alert.displayName = "Alert"

const AlertTitle = React.forwardRef(({ className, ...props }, ref) => (
  <h5
    ref={ref}
    className={cn("mb-1 font-medium leading-none tracking-tight", className)}
    {...props}
  />
))
AlertTitle.displayName = "AlertTitle"

const AlertDescription = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("text-sm [&_p]:leading-relaxed", className)}
    {...props}
  />
))
AlertDescription.displayName = "AlertDescription"

// Helper component with icons
const AlertWithIcon = ({ variant = "info", title, children, className, ...props }) => {
  const icons = {
    info: Info,
    success: CheckCircle,
    warning: AlertTriangle,
    destructive: AlertCircle,
    default: Info,
  }
  
  const Icon = icons[variant]
  
  return (
    <Alert variant={variant} className={className} {...props}>
      <Icon className="h-4 w-4" />
      {title && <AlertTitle>{title}</AlertTitle>}
      <AlertDescription>{children}</AlertDescription>
    </Alert>
  )
}

export { Alert, AlertTitle, AlertDescription, AlertWithIcon }