import * as Menu from "@radix-ui/react-dropdown-menu";
import * as React from "react";
import { cn } from "@/lib/utils";

export const DropdownMenu = Menu.Root;
export const DropdownMenuTrigger = Menu.Trigger;
export function DropdownMenuContent({ className, align = "end", ...props }: React.ComponentProps<typeof Menu.Content>) {
  return (
    <Menu.Portal>
      <Menu.Content align={align} sideOffset={6}
        className={cn("z-50 min-w-44 rounded-lg border bg-card p-1 shadow-lg animate-fade-in", className)} {...props} />
    </Menu.Portal>
  );
}
export function DropdownMenuItem({ className, danger, ...props }: React.ComponentProps<typeof Menu.Item> & { danger?: boolean }) {
  return (
    <Menu.Item className={cn("flex cursor-pointer items-center gap-2 rounded-md px-2.5 py-1.5 text-sm outline-none " +
      "data-[highlighted]:bg-muted data-[disabled]:opacity-50 [&_svg]:size-4",
      danger && "text-danger", className)} {...props} />
  );
}
export const DropdownMenuSeparator = () => <Menu.Separator className="my-1 h-px bg-border" />;
