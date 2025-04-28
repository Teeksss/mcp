import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  TextField,
  Button,
  CircularProgress,
  Typography,
  Chip,
  Select,
  MenuItem,
  Alert,
} from '@mui/material';
import { useQuery, useMutation } from '@tanstack/react-query';
import { ModelService } from '../services/ModelService';
import { useAuth } from '../hooks/useAuth';

interface ModelInterfaceProps {
  onResult: (result: string) => void;
}

export const ModelInterface: React.FC<ModelInterfaceProps> = ({ onResult }) => {
  const { user } = useAuth();
  const [prompt, setPrompt] = useState('');
  const [selectedModel, setSelectedModel] = useState('llama');
  const [useRag, setUseRag] = useState(false);

  const { data: models, isLoading: modelsLoading } = useQuery(
    ['available-models'],
    ModelService.getAvailableModels
  );

  const generateMutation = useMutation(
    (data: { prompt: string; model: string; useRag: boolean }) =>
      ModelService.generateResponse(data),
    {
      onSuccess: (data) => {
        onResult(data.response);
      },
    }
  );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt) return;

    generateMutation.mutate({
      prompt,
      model: selectedModel,
      useRag,
    });
  };

  return (
    <Paper elevation={3} sx={{ p: 3, my: 2 }}>
      <Box component="form" onSubmit={handleSubmit}>
        <Typography variant="h6" gutterBottom>
          Model Interface
        </Typography>
        
        <Select
          fullWidth
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          sx={{ mb: 2 }}
        >
          {models?.map((model) => (
            <MenuItem key={model.id} value={model.id}>
              {model.name} - {model.description}
            </MenuItem>
          ))}
        </Select>

        <TextField
          fullWidth
          multiline
          rows={4}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter your prompt here..."
          sx={{ mb: 2 }}
        />

        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          <Chip
            label="Use RAG"
            clickable
            color={useRag ? 'primary' : 'default'}
            onClick={() => setUseRag(!useRag)}
          />
          {user?.role === 'admin' && (
            <Chip
              label="Advanced Options"
              clickable
              onClick={() => {/* Show advanced options */}}
            />
          )}
        </Box>

        <Button
          fullWidth
          variant="contained"
          type="submit"
          disabled={generateMutation.isLoading || !prompt}
        >
          {generateMutation.isLoading ? (
            <CircularProgress size={24} />
          ) : (
            'Generate Response'
          )}
        </Button>

        {generateMutation.error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {(generateMutation.error as Error).message}
          </Alert>
        )}
      </Box>
    </Paper>
  );
};