import React, { useState } from 'react';

interface TiemposData {
  planning_time_ms: number;
  execution_time_ms: number;
}

const Tiempos: React.FC = () => {
  const [data, setData] = useState<TiemposData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const fetchData = async () => {
    setLoading(true);
    setError(null); 
    try {
      const response = await fetch("http://localhost:8000/get_time"); 
      if (!response.ok) {
        throw new Error("Error en la respuesta de la API");
      }
      const result: TiemposData = await response.json();
      setData(result);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Error desconocido";
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section>
      <button onClick={fetchData} disabled={loading} className='bg-white p-2 rounded-lg shadow-md mb-3 font-light'>
        Ejecutar Consulta
      </button>
      <h1>Tiempos de Ejecución de la Consulta</h1>
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}
      {data && (
        <div>
          <p><strong>Planning Time:</strong> {data.planning_time_ms} ms</p>
          <p><strong>Execution Time:</strong> {data.execution_time_ms} ms</p>
        </div>
      )}
    </section>
  );
};

export default Tiempos;
