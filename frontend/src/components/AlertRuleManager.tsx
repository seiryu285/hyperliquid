import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControl,
  Grid,
  IconButton,
  InputLabel,
  MenuItem,
  Select,
  TextField,
  Typography
} from '@mui/material';
import { DataGrid, GridColDef } from '@mui/x-data-grid';
import { Edit, Delete, Add } from '@mui/icons-material';
import { AlertRule, NotificationChannel } from '../types/alerts';
import { useAlertRules } from '../hooks/useAlertRules';
import { useNotify } from '../hooks/useNotify';

const AlertRuleManager: React.FC = () => {
  const [rules, setRules] = useState<AlertRule[]>([]);
  const [channels, setChannels] = useState<NotificationChannel[]>([]);
  const [openDialog, setOpenDialog] = useState(false);
  const [editingRule, setEditingRule] = useState<AlertRule | null>(null);
  const { getRules, createRule, updateRule, deleteRule } = useAlertRules();
  const { notify } = useNotify();

  // Load rules and channels
  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [rulesData, channelsData] = await Promise.all([
        getRules(),
        fetch('/api/alert-config/channels').then(res => res.json())
      ]);
      setRules(rulesData);
      setChannels(channelsData);
    } catch (error) {
      notify('Error loading data', 'error');
    }
  };

  // Column definitions
  const columns: GridColDef[] = [
    { field: 'name', headerName: 'Name', flex: 1 },
    { field: 'description', headerName: 'Description', flex: 2 },
    { field: 'event_type', headerName: 'Event Type', flex: 1 },
    { field: 'threshold', headerName: 'Threshold', width: 100 },
    {
      field: 'actions',
      headerName: 'Actions',
      width: 120,
      renderCell: (params) => (
        <Box>
          <IconButton
            size="small"
            onClick={() => handleEdit(params.row)}
          >
            <Edit />
          </IconButton>
          <IconButton
            size="small"
            onClick={() => handleDelete(params.row.id)}
          >
            <Delete />
          </IconButton>
        </Box>
      )
    }
  ];

  // Dialog form state
  const [formData, setFormData] = useState<Partial<AlertRule>>({
    name: '',
    description: '',
    event_type: '',
    threshold: 0,
    window_minutes: 5,
    channels: []
  });

  const handleEdit = (rule: AlertRule) => {
    setEditingRule(rule);
    setFormData(rule);
    setOpenDialog(true);
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteRule(id);
      notify('Rule deleted successfully', 'success');
      loadData();
    } catch (error) {
      notify('Error deleting rule', 'error');
    }
  };

  const handleSubmit = async () => {
    try {
      if (editingRule) {
        await updateRule(editingRule.id, formData as AlertRule);
        notify('Rule updated successfully', 'success');
      } else {
        await createRule(formData as AlertRule);
        notify('Rule created successfully', 'success');
      }
      setOpenDialog(false);
      loadData();
    } catch (error) {
      notify('Error saving rule', 'error');
    }
  };

  return (
    <Box sx={{ height: '100%', width: '100%', p: 2 }}>
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <Box display="flex" justifyContent="space-between" mb={2}>
            <Typography variant="h5">Alert Rules</Typography>
            <Button
              variant="contained"
              startIcon={<Add />}
              onClick={() => {
                setEditingRule(null);
                setFormData({
                  name: '',
                  description: '',
                  event_type: '',
                  threshold: 0,
                  window_minutes: 5,
                  channels: []
                });
                setOpenDialog(true);
              }}
            >
              Add Rule
            </Button>
          </Box>
        </Grid>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <DataGrid
                rows={rules}
                columns={columns}
                pageSize={10}
                rowsPerPageOptions={[10]}
                autoHeight
                disableSelectionOnClick
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Dialog
        open={openDialog}
        onClose={() => setOpenDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {editingRule ? 'Edit Alert Rule' : 'Create Alert Rule'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Name"
                value={formData.name}
                onChange={(e) => setFormData({
                  ...formData,
                  name: e.target.value
                })}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Description"
                value={formData.description}
                onChange={(e) => setFormData({
                  ...formData,
                  description: e.target.value
                })}
              />
            </Grid>
            <Grid item xs={6}>
              <FormControl fullWidth>
                <InputLabel>Event Type</InputLabel>
                <Select
                  value={formData.event_type}
                  label="Event Type"
                  onChange={(e) => setFormData({
                    ...formData,
                    event_type: e.target.value
                  })}
                >
                  <MenuItem value="auth_failure">Authentication Failure</MenuItem>
                  <MenuItem value="rate_limit">Rate Limit</MenuItem>
                  <MenuItem value="brute_force">Brute Force</MenuItem>
                  <MenuItem value="anomaly">Anomaly</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={3}>
              <TextField
                fullWidth
                type="number"
                label="Threshold"
                value={formData.threshold}
                onChange={(e) => setFormData({
                  ...formData,
                  threshold: parseInt(e.target.value)
                })}
              />
            </Grid>
            <Grid item xs={3}>
              <TextField
                fullWidth
                type="number"
                label="Window (minutes)"
                value={formData.window_minutes}
                onChange={(e) => setFormData({
                  ...formData,
                  window_minutes: parseInt(e.target.value)
                })}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Notification Channels</InputLabel>
                <Select
                  multiple
                  value={formData.channels || []}
                  label="Notification Channels"
                  onChange={(e) => setFormData({
                    ...formData,
                    channels: e.target.value as string[]
                  })}
                >
                  {channels.map((channel) => (
                    <MenuItem key={channel.name} value={channel.name}>
                      {channel.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleSubmit}
            disabled={!formData.name || !formData.event_type}
          >
            {editingRule ? 'Update' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AlertRuleManager;
