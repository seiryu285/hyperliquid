import React, { useState } from 'react';
import {
    Box,
    Button,
    TextField,
    Typography,
    Container,
    Alert,
    CircularProgress,
    Paper
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { authService } from '../../services/auth';

const VerifyContainer = styled(Container)`
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-top: ${({ theme }) => theme.spacing(8)};
`;

const Form = styled('form')`
    width: 100%;
    max-width: 360px;
    margin-top: ${({ theme }) => theme.spacing(1)};
`;

interface TwoFactorVerifyProps {
    onVerified: () => void;
    onCancel: () => void;
}

export const TwoFactorVerify: React.FC<TwoFactorVerifyProps> = ({
    onVerified,
    onCancel
}) => {
    const [verificationCode, setVerificationCode] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            await authService.verify2FA(verificationCode);
            onVerified();
        } catch (err) {
            setError('Invalid verification code. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <VerifyContainer maxWidth="sm">
            <Paper elevation={2} sx={{ p: 4, width: '100%', maxWidth: 400 }}>
                <Typography component="h1" variant="h5" align="center" gutterBottom>
                    Two-Factor Authentication
                </Typography>

                <Typography variant="body1" align="center" paragraph>
                    Enter the verification code from your authenticator app to continue.
                </Typography>

                {error && (
                    <Alert severity="error" sx={{ mb: 2 }}>
                        {error}
                    </Alert>
                )}

                <Form onSubmit={handleSubmit}>
                    <TextField
                        variant="outlined"
                        margin="normal"
                        required
                        fullWidth
                        id="verificationCode"
                        label="Verification Code"
                        name="verificationCode"
                        autoComplete="off"
                        autoFocus
                        value={verificationCode}
                        onChange={(e) => setVerificationCode(e.target.value)}
                        disabled={loading}
                    />

                    <Button
                        type="submit"
                        fullWidth
                        variant="contained"
                        color="primary"
                        disabled={loading}
                        sx={{ mt: 2 }}
                    >
                        {loading ? (
                            <CircularProgress size={24} color="inherit" />
                        ) : (
                            'Verify'
                        )}
                    </Button>

                    <Button
                        fullWidth
                        color="inherit"
                        onClick={onCancel}
                        disabled={loading}
                        sx={{ mt: 1 }}
                    >
                        Cancel
                    </Button>
                </Form>

                <Box mt={3}>
                    <Typography variant="body2" color="textSecondary" align="center">
                        Lost access to your authenticator app?{' '}
                        <Button
                            color="primary"
                            onClick={() => {/* TODO: Implement backup code flow */}}
                        >
                            Use a backup code
                        </Button>
                    </Typography>
                </Box>
            </Paper>
        </VerifyContainer>
    );
};
