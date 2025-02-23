import React, { useState } from 'react';
import {
    Box,
    Button,
    TextField,
    Typography,
    Container,
    Alert,
    CircularProgress,
    Paper,
    List,
    ListItem,
    ListItemText,
    Divider
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { authService } from '../../services/auth';

const SetupContainer = styled(Container)`
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-top: ${({ theme }) => theme.spacing(4)};
`;

const QRCode = styled('img')`
    width: 200px;
    height: 200px;
    margin: ${({ theme }) => theme.spacing(2)};
`;

const BackupCodesList = styled(List)`
    width: 100%;
    max-width: 360px;
    background-color: ${({ theme }) => theme.palette.background.paper};
    margin: ${({ theme }) => theme.spacing(2, 0)};
`;

const Form = styled('form')`
    width: 100%;
    max-width: 360px;
    margin-top: ${({ theme }) => theme.spacing(2)};
`;

interface TwoFactorSetupProps {
    onComplete: () => void;
}

export const TwoFactorSetup: React.FC<TwoFactorSetupProps> = ({ onComplete }) => {
    const [setupData, setSetupData] = useState<{
        secret?: string;
        qrCode?: string;
        backupCodes?: string[];
    }>({});
    const [verificationCode, setVerificationCode] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const [step, setStep] = useState<'setup' | 'verify'>('setup');

    const handleSetup = async () => {
        setError('');
        setLoading(true);

        try {
            const response = await authService.setup2FA();
            setSetupData(response);
            setStep('verify');
        } catch (err) {
            setError('Failed to set up 2FA. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const handleVerify = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            await authService.verify2FA(verificationCode);
            onComplete();
        } catch (err) {
            setError('Invalid verification code. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const copyToClipboard = (text: string) => {
        navigator.clipboard.writeText(text);
    };

    if (step === 'setup') {
        return (
            <SetupContainer maxWidth="sm">
                <Typography component="h1" variant="h5" gutterBottom>
                    Set up Two-Factor Authentication
                </Typography>

                <Typography variant="body1" align="center" paragraph>
                    Two-factor authentication adds an extra layer of security to your
                    account. Once enabled, you'll need to enter a verification code
                    from your authenticator app when signing in.
                </Typography>

                <Button
                    variant="contained"
                    color="primary"
                    onClick={handleSetup}
                    disabled={loading}
                >
                    {loading ? (
                        <CircularProgress size={24} color="inherit" />
                    ) : (
                        'Begin Setup'
                    )}
                </Button>

                {error && (
                    <Alert severity="error" sx={{ mt: 2 }}>
                        {error}
                    </Alert>
                )}
            </SetupContainer>
        );
    }

    return (
        <SetupContainer maxWidth="sm">
            <Typography component="h1" variant="h5" gutterBottom>
                Complete 2FA Setup
            </Typography>

            {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                </Alert>
            )}

            <Paper elevation={2} sx={{ p: 3, width: '100%', maxWidth: 400 }}>
                <Typography variant="h6" gutterBottom>
                    1. Scan QR Code
                </Typography>

                <Typography variant="body2" paragraph>
                    Scan this QR code with your authenticator app:
                </Typography>

                {setupData.qrCode && (
                    <Box display="flex" justifyContent="center">
                        <QRCode
                            src={`data:image/png;base64,${setupData.qrCode}`}
                            alt="2FA QR Code"
                        />
                    </Box>
                )}

                <Typography variant="body2" paragraph>
                    Or manually enter this code:
                </Typography>

                <Typography
                    variant="mono"
                    component="div"
                    sx={{
                        backgroundColor: (theme) => theme.palette.grey[100],
                        padding: 1,
                        borderRadius: 1,
                        fontFamily: 'monospace',
                        cursor: 'pointer'
                    }}
                    onClick={() => copyToClipboard(setupData.secret || '')}
                >
                    {setupData.secret}
                </Typography>

                <Typography variant="h6" sx={{ mt: 3 }} gutterBottom>
                    2. Save Backup Codes
                </Typography>

                <Typography variant="body2" paragraph>
                    Save these backup codes in a secure place. You can use them to
                    access your account if you lose your authenticator device:
                </Typography>

                <BackupCodesList>
                    {setupData.backupCodes?.map((code, index) => (
                        <React.Fragment key={code}>
                            <ListItem
                                button
                                onClick={() => copyToClipboard(code)}
                            >
                                <ListItemText
                                    primary={code}
                                    secondary="Click to copy"
                                />
                            </ListItem>
                            {index < (setupData.backupCodes?.length || 0) - 1 && (
                                <Divider />
                            )}
                        </React.Fragment>
                    ))}
                </BackupCodesList>

                <Typography variant="h6" sx={{ mt: 3 }} gutterBottom>
                    3. Verify Setup
                </Typography>

                <Form onSubmit={handleVerify}>
                    <TextField
                        variant="outlined"
                        margin="normal"
                        required
                        fullWidth
                        id="verificationCode"
                        label="Verification Code"
                        name="verificationCode"
                        autoComplete="off"
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
                            'Verify and Enable 2FA'
                        )}
                    </Button>
                </Form>
            </Paper>
        </SetupContainer>
    );
};
