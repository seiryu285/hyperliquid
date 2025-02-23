import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    Box,
    Button,
    TextField,
    Typography,
    Container,
    Alert,
    CircularProgress
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { authService, LoginCredentials } from '../../services/auth';

const FormContainer = styled(Container)`
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-top: ${({ theme }) => theme.spacing(8)};
`;

const Form = styled('form')`
    width: 100%;
    margin-top: ${({ theme }) => theme.spacing(1)};
`;

const SubmitButton = styled(Button)`
    margin: ${({ theme }) => theme.spacing(3, 0, 2)};
`;

export const LoginForm: React.FC = () => {
    const navigate = useNavigate();
    const [credentials, setCredentials] = useState<LoginCredentials>({
        email: '',
        password: ''
    });
    const [error, setError] = useState<string>('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            await authService.login(credentials);
            navigate('/dashboard');
        } catch (err) {
            setError('Invalid email or password');
        } finally {
            setLoading(false);
        }
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setCredentials({
            ...credentials,
            [e.target.name]: e.target.value
        });
    };

    return (
        <FormContainer maxWidth="xs">
            <Typography component="h1" variant="h5">
                Sign in to HyperLiquid
            </Typography>

            <Form onSubmit={handleSubmit}>
                {error && (
                    <Alert severity="error" sx={{ mb: 2 }}>
                        {error}
                    </Alert>
                )}

                <TextField
                    variant="outlined"
                    margin="normal"
                    required
                    fullWidth
                    id="email"
                    label="Email Address"
                    name="email"
                    autoComplete="email"
                    autoFocus
                    value={credentials.email}
                    onChange={handleChange}
                    disabled={loading}
                />

                <TextField
                    variant="outlined"
                    margin="normal"
                    required
                    fullWidth
                    name="password"
                    label="Password"
                    type="password"
                    id="password"
                    autoComplete="current-password"
                    value={credentials.password}
                    onChange={handleChange}
                    disabled={loading}
                />

                <SubmitButton
                    type="submit"
                    fullWidth
                    variant="contained"
                    color="primary"
                    disabled={loading}
                >
                    {loading ? (
                        <CircularProgress size={24} color="inherit" />
                    ) : (
                        'Sign In'
                    )}
                </SubmitButton>

                <Box mt={2}>
                    <Button
                        fullWidth
                        color="primary"
                        onClick={() => navigate('/register')}
                        disabled={loading}
                    >
                        Don't have an account? Sign Up
                    </Button>
                </Box>
            </Form>
        </FormContainer>
    );
};
