interface Props {
  data: number[];
  width?: number;
  height?: number;
}

export default function LiveEquityCurve({ data, width = 500, height = 80 }: Props) {
  if (data.length < 2) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const initial = data[0];
  const current = data[data.length - 1];
  const positive = current >= initial;

  const points = data.map((value, idx) => {
    const x = (idx / (data.length - 1)) * width;
    const y = height - 4 - ((value - min) / range) * (height - 8);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const pathD = `M ${points.join(' L ')}`;
  const strokeColor = positive ? 'rgb(16, 185, 129)' : 'rgb(239, 68, 68)';

  // Fill area under curve
  const fillPoints = `M 0,${height} L ${points.join(' L ')} L ${width},${height} Z`;
  const fillColor = positive ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)';

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="mt-2">
      <path d={fillPoints} fill={fillColor} />
      <path d={pathD} fill="none" stroke={strokeColor} strokeWidth={1.5} />
    </svg>
  );
}
