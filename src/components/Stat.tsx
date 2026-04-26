/** Single label + value pair used inside <StatsBar />. */
export function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <dt className="text-zinc-500">{label}</dt>
      <dd className="font-mono">{value}</dd>
    </div>
  );
}
