import { Tooltip } from "./ui/tooltip";

/** Single-series column chart of counts per year (one hue, rounded data-ends, per-bar tooltip, table fallback). */
export function YearBars({ data, label, height = 120 }: { data: { year: number; count: number }[]; label: string; height?: number }) {
  const max = Math.max(1, ...data.map((d) => d.count));
  return (
    <figure className="w-full">
      <div className="flex items-end gap-[2px]" style={{ height }} role="img" aria-label={`${label}: bar chart by year`}>
        {data.map((d) => (
          <Tooltip key={d.year} content={<span className="tabular-nums">{d.year}: {d.count.toLocaleString()} {label.toLowerCase()}</span>}>
            {/* Hit target spans the full column height; the visible mark is the inner bar. */}
            <div className="group flex h-full min-w-0 flex-1 cursor-default items-end" tabIndex={0} aria-label={`${d.year}: ${d.count}`}>
              <div className="w-full rounded-t-[4px] bg-chart-1 transition-opacity group-hover:opacity-80 group-focus-visible:opacity-80"
                style={{ height: `${Math.max(d.count ? 2 : 0, (d.count / max) * 100)}%` }} />
            </div>
          </Tooltip>
        ))}
      </div>
      <div className="mt-1 flex gap-[2px] border-t pt-1 text-[10px] tabular-nums text-muted-foreground">
        {data.map((d, i) => (
          <span key={d.year} className="min-w-0 flex-1 text-center">{i % 2 === 0 || i === data.length - 1 ? `'${String(d.year).slice(2)}` : ""}</span>
        ))}
      </div>
      <table className="sr-only">
        <caption>{label} per year</caption>
        <thead><tr><th>Year</th><th>{label}</th></tr></thead>
        <tbody>{data.map((d) => <tr key={d.year}><td>{d.year}</td><td>{d.count}</td></tr>)}</tbody>
      </table>
    </figure>
  );
}
