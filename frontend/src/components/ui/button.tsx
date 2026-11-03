import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { Loader2 } from "lucide-react";
import * as React from "react";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors " +
    "disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-4 [&_svg]:shrink-0 cursor-pointer",
  {
    variants: {
      variant: {
        primary: "bg-primary text-primary-foreground shadow-sm hover:bg-primary/90",
        secondary: "bg-muted text-foreground hover:bg-muted/70 border border-border",
        outline: "border border-input bg-card hover:bg-muted",
        ghost: "hover:bg-muted text-foreground",
        danger: "bg-danger text-white hover:bg-danger/90 dark:text-background",  // dark: 6.6:1 (white would be 2.9:1)
        link: "text-primary underline-offset-4 hover:underline px-0",
      },
      size: { sm: "h-8 px-3 text-xs", md: "h-9 px-4", lg: "h-10 px-5", icon: "h-9 w-9", "icon-sm": "h-7 w-7" },
    },
    defaultVariants: { variant: "primary", size: "md" },
  },
);

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {
  asChild?: boolean;
  loading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild, loading, children, disabled, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp ref={ref} className={cn(buttonVariants({ variant, size }), className)} disabled={disabled || loading} {...props}>
        {asChild ? children : (<>{loading && <Loader2 className="animate-spin" />}{children}</>)}
      </Comp>
    );
  },
);
Button.displayName = "Button";
