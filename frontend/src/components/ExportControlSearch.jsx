import React, { useState } from 'react';
import styled from 'styled-components';
import ReactMarkdown from 'react-markdown';

const Container = styled.div`
  max-width: 800px;
  margin: 2rem auto;
  padding: 0 1rem;
`;

const SearchContainer = styled.div`
  display: flex;
  gap: 1rem;
  margin-bottom: 2rem;
`;

const Input = styled.input`
  flex: 1;
  padding: 0.5rem;
  font-size: 1rem;
  border: 1px solid #ccc;
  border-radius: 4px;
`;

const Button = styled.button`
  padding: 0.5rem 1rem;
  font-size: 1rem;
  background-color: #0066cc;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  
  &:hover {
    background-color: #0052a3;
  }
  
  &:disabled {
    background-color: #cccccc;
    cursor: not-allowed;
  }
`;

const ResultContainer = styled.div`
  padding: 1rem;
  border: 1px solid #eee;
  border-radius: 4px;
  background-color: #f9f9f9;
`;

const ExportControlSearch = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
        // const response = await fetch(`http://localhost:8000/get_part_details?search_query=${encodeURIComponent(searchQuery)}`);
      // dummy response markdown formatted
      // const response = "## Part Details\n\n- **Part Number:** 123456\n- **Description:** This is a test part\n- **Export Control Classification:** Class 1\n- **Export Control Classification Number:** 1234567890\n- **Export Control Classification Notes:** This is a test part";
      const response = await fetch(`https://vcm-45281.vm.duke.edu:8000/get_part_details?search_query=${encodeURIComponent(searchQuery)}`);


        if (!response.ok) {
          throw new Error('Failed to fetch part details');
        }

        const data = await response.json();
      setResult(data.result);
    } catch (err) {
      setError('An error occurred while fetching the results. Please try again.');
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container>
      <h1>Export Control Classification Search</h1>
      
      <SearchContainer>
        <Input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Enter part number or description..."
          disabled={isLoading}
        />
        <Button 
          onClick={handleSearch}
          disabled={isLoading || !searchQuery.trim()}
        >
          {isLoading ? 'Searching...' : 'Search'}
        </Button>
      </SearchContainer>

      {error && (
        <div style={{ color: 'red', marginBottom: '1rem' }}>
          {error}
        </div>
      )}

      {result && (
        <ResultContainer>
          <ReactMarkdown>{result}</ReactMarkdown>
        </ResultContainer>
      )}
    </Container>
  );
};

export default ExportControlSearch; 
