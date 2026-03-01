import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface Props {
  data: Record<string, unknown>[];
}

export default function VisionVsStateChart({ data }: Props) {
  const chartData = data
    .map((d, i) => {
      if (!d) return null;
      return {
        frame: `Frame ${i}`,
        vision: Math.round(((d.vision_share as number) || 0) * 100),
        state: Math.round((1 - ((d.vision_share as number) || 0)) * 100),
      };
    })
    .filter(Boolean);

  if (chartData.length === 0) {
    return <p style={{ color: "var(--text-body)", fontSize: 13 }}>No data</p>;
  }

  return (
    <div style={{ width: "100%", maxWidth: 700 }}>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} stackOffset="expand">
          <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
          <XAxis dataKey="frame" fontSize={11} />
          <YAxis
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
            fontSize={11}
          />
          <Tooltip
            formatter={(value: number | undefined) => `${value ?? 0}%`}
          />
          <Legend />
          <Bar dataKey="vision" stackId="a" fill="#3B3BD3" name="Vision" />
          <Bar dataKey="state" stackId="a" fill="#7F8385" name="State" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
