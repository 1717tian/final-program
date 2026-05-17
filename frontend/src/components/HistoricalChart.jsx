import React from 'react';

export default function HistoricalChart({ data, dates }) {
  return (
    <div>
      <h2>历史拟合</h2>
      <ul>
        {dates && data && dates.map((d,i)=>(
          <li key={i}>{d}: {data[i]}</li>
        ))}
      </ul>
    </div>
  );
}