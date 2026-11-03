import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { YearBars } from "@/components/YearBars";
import { TooltipProvider } from "@/components/ui/tooltip";
import type { Gap } from "@/lib/types";
import { CheckBadge } from "./research/GapCard";

describe("YearBars", () => {
  it("renders one bar per year and an accessible table", () => {
    render(<TooltipProvider><YearBars label="Publications" data={[{ year: 2024, count: 10 }, { year: 2025, count: 20 }]} /></TooltipProvider>);
    expect(screen.getByRole("img", { name: /Publications/ })).toBeInTheDocument();
    expect(screen.getByLabelText("2025: 20")).toBeInTheDocument();
    expect(screen.getByRole("table")).toHaveTextContent("2024");
  });
});

describe("CheckBadge", () => {
  const gap = (s?: Gap["verification_status"]) => ({ verification_status: s }) as Gap;
  it("labels the duplication-test outcome", () => {
    const { rerender } = render(<CheckBadge gap={gap("verified_open")} />);
    expect(screen.getByText("Checked: still open")).toBeInTheDocument();
    rerender(<CheckBadge gap={gap("addressed")} />);
    expect(screen.getByText("Checked: addressed elsewhere")).toBeInTheDocument();
    rerender(<CheckBadge gap={gap(undefined)} />);
    expect(screen.getByText("Not yet checked")).toBeInTheDocument();
  });
});
