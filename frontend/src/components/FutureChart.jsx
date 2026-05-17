import React from 'react';

export default function FutureChart({ data, dates, alerts }) {
  return (
    <div>
      <h2>未来预测</h2>
      <ul>
        {dates && data && dates.map((d,i)=>(
          <li key={i}>{d}: {data[i]} ({alerts[i]})</li>
        ))}
      </ul>
    </div>
  );
}