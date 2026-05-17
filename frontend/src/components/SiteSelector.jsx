import React from 'react';

export default function SiteSelector({ stations, station, setStation }) {
  return (
    <select value={station} onChange={e=>setStation(e.target.value)}>
      {stations.map((s, idx)=>(<option key={idx} value={s}>{s}</option>))}
    </select>
  );
}