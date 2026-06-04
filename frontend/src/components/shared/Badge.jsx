export default function Badge({ text, color }) {
  return (
    <span className="badge" style={{ '--badge-color': color }}>
      {text}
    </span>
  )
}
