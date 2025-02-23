import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
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
import { authService } from '../../services/auth';

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

interface ResetFormData {
    email?: string;
    token?: string;
    password?: string;
    confirmPassword?: string;
}

export const PasswordResetForm: React.FC = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const [formData, setFormData] = useState<ResetFormData>({});
    const [error, setError] = useState<string>('');
    const [success, setSuccess] = useState<string>('');
    const [loading, setLoading] = useState(false);

    // Get token from URL if present
    const token = new URLSearchParams(location.search).get('token');
    const isConfirmReset = !!token;

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setSuccess('');
        setLoading(true);

        try {
            if (isConfirmReset) {
                // Confirm password reset
                if (!formData.password || !formData.confirmPassword) {
                    throw new Error('Please enter both password fields');
                }
                if (formData.password !== formData.confirmPassword) {
                    throw new Error('Passwords do not match');
                }

                await authService.confirmPasswordReset(token, formData.password);
                setSuccess('Password has been reset successfully');
                setTimeout(() => navigate('/login'), 3000);
            } else {
                // Request password reset
                if (!formData.email) {
                    throw new Error('Please enter your email');
                }

                await authService.requestPasswordReset(formData.email);
                setSuccess(
                    'If an account exists with this email, ' +
                    'you will receive password reset instructions'
                );
            }
        } catch (err) {
            setError(err.message || 'An error occurred');
        } finally {
            setLoading(false);
        }
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    return (
        <FormContainer maxWidth="xs">
            <Typography component="h1" variant="h5">
                {isConfirmReset ? 'Reset Password' : 'Request Password Reset'}
            </Typography>

            <Form onSubmit={handleSubmit}>
                {error && (
                    <Alert severity="error" sx={{ mb: 2 }}>
                        {error}
                    </Alert>
                )}

                {success && (
                    <Alert severity="success" sx={{ mb: 2 }}>
                        {success}
                    </Alert>
                )}

                {!isConfirmReset ? (
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
                        value={formData.email || ''}
                        onChange={handleChange}
                        disabled={loading}
                    />
                ) : (
                    <>
                        <TextField
                            variant="outlined"
                            margin="normal"
                            required
                            fullWidth
                            name="password"
                            label="New Password"
                            type="password"
                            id="password"
                            value={formData.password || ''}
                            onChange={handleChange}
                            disabled={loading}
                        />
                        <TextField
                            variant="outlined"
                            margin="normal"
                            required
                            fullWidth
                            name="confirmPassword"
                            label="Confirm New Password"
                            type="password"
                            id="confirmPassword"
                            value={formData.confirmPassword || ''}
                            onChange={handleChange}
                            disabled={loading}
                        />
                    </>
                )}

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
                        isConfirmReset ? 'Reset Password' : 'Send Reset Link'
                    )}
                </SubmitButton>

                <Box mt={2}>
                    <Button
                        fullWidth
                        color="primary"
                        onClick={() => navigate('/login')}
                        disabled={loading}
                    >
                        Back to Login
                    </Button>
                </Box>
            </Form>
        </FormContainer>
    );
};
